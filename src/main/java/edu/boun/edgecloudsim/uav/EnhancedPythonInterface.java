package edu.boun.edgecloudsim.uav;

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger; 
import org.json.*;
import java.util.logging.*;

/**
 * 增强型Python接口类，提供与Python RL服务器的通信功能
 * 修复版本 - 解决连接不稳定和训练请求阻塞问题
 */
public class EnhancedPythonInterface {
    private Socket socket;
    private BufferedReader in;
    private PrintWriter out;
    private String host;
    private int port;
    private volatile boolean connected;
    private ExecutorService executor;
    private Logger logger;
    private ConcurrentHashMap<String, Object> statistics;
    private final Object lock = new Object();
    private int reconnectAttempts;
    private final int MAX_RECONNECT_ATTEMPTS = 10;
    // 降低超时时间，避免长时间阻塞
    private final int READ_TIMEOUT_MS = 10000; // 降低到10秒
    private final int CONNECTION_TIMEOUT_MS = 5000; // 降低到5秒
    private final int READ_BUFFER_SIZE = 8192;
    // 添加重连延迟常量
    private final long RECONNECT_DELAY_MS = 1000; // 1秒基础重连延迟

    // 固定状态和动作维度以匹配Python服务器
    private final int STATE_DIMENSION = 12;
    private final int ACTION_DIMENSION = 3;
    
    // 支持的算法类型
    public static final String ALGORITHM_PRIORITIZED_DDPG = "prioritized_ddpg";
    public static final String ALGORITHM_DDPG = "ddpg";
    public static final String ALGORITHM_DQN = "dqn";
    public static final String ALGORITHM_PPO = "ppo";
    
    // 默认算法类型
    private String defaultAlgorithm = ALGORITHM_PRIORITIZED_DDPG;
    
    // 用于管理请求和响应的映射
    private final ConcurrentHashMap<String, CompletableFuture<JSONObject>> pendingRequests = new ConcurrentHashMap<>();
    private final ScheduledExecutorService requestTimeoutScheduler = Executors.newSingleThreadScheduledExecutor();
    
    // 响应读取线程
    private Thread responseReaderThread;
    private volatile boolean stopReading = false;
    
    // 连接保护
    private final Semaphore connectionSemaphore = new Semaphore(1);
    private final AtomicInteger consecutiveFailures = new AtomicInteger(0);
    private final int MAX_CONSECUTIVE_FAILURES = 50;
    private long lastConnectionAttempt = 0;
    private final long CONNECTION_COOLDOWN_MS = 5000;
    
    // 请求队列和批处理
    private final LinkedBlockingQueue<PendingRequest> requestQueue = new LinkedBlockingQueue<>();
    private volatile boolean processingQueue = false;
    private final int BATCH_SIZE = 10;
    private final int QUEUE_PROCESS_INTERVAL_MS = 100;
    
    // 新增消息边界定义
    private final String MESSAGE_BOUNDARY = "\n\n";

    /**
     * 构造函数
     * @param host Python服务器主机名
     * @param port Python服务器端口
     */
    public EnhancedPythonInterface(String host, int port) {
        this.host = host;
        this.port = port;
        this.connected = false;
        this.executor = Executors.newCachedThreadPool();
        this.logger = Logger.getLogger(EnhancedPythonInterface.class.getName());
        this.statistics = new ConcurrentHashMap<>();
        this.reconnectAttempts = 0;
        
        // 设置日志
        try {
            FileHandler fileHandler = new FileHandler("enhanced_python_interface.log", true);
            fileHandler.setFormatter(new SimpleFormatter());
            this.logger.addHandler(fileHandler);
            this.logger.setLevel(Level.INFO);
            
            // 添加控制台输出，便于调试
            ConsoleHandler consoleHandler = new ConsoleHandler();
            consoleHandler.setFormatter(new SimpleFormatter());
            this.logger.addHandler(consoleHandler);
        } catch (IOException e) {
            System.err.println("无法设置日志文件: " + e.getMessage());
        }
        
        logger.info("初始化接口，使用固定状态维度=" + STATE_DIMENSION + ", 动作维度=" + ACTION_DIMENSION);
        logger.info("默认使用算法: " + defaultAlgorithm);
        
        // 启动请求队列处理线程
        startQueueProcessor();
    }

    /**
     * 设置默认算法
     * @param algorithm 要设置的默认算法
     */
    public void setDefaultAlgorithm(String algorithm) {
        if (isValidAlgorithm(algorithm)) {
            this.defaultAlgorithm = algorithm;
            logger.info("设置默认算法为: " + algorithm);
        } else {
            logger.warning("尝试设置无效的算法: " + algorithm + "，维持默认算法: " + defaultAlgorithm);
        }
    }
    
    /**
     * 获取当前默认算法
     * @return 当前使用的默认算法
     */
    public String getDefaultAlgorithm() {
        return defaultAlgorithm;
    }
    
    /**
     * 验证算法是否有效
     * @param algorithm 要验证的算法名称
     * @return 是否是有效的算法
     */
    private boolean isValidAlgorithm(String algorithm) {
        return algorithm != null && (
            algorithm.equals(ALGORITHM_PRIORITIZED_DDPG) ||
            algorithm.equals(ALGORITHM_DDPG) ||
            algorithm.equals(ALGORITHM_DQN) ||
            algorithm.equals(ALGORITHM_PPO)
        );
    }

    /**
     * 启动请求队列处理器
     */
    private void startQueueProcessor() {
        processingQueue = true;
        executor.submit(() -> {
            while (processingQueue && !Thread.currentThread().isInterrupted()) {
                try {
                    // 批量处理请求
                    processBatchRequests();
                    
                    // 短暂休眠以避免CPU空转
                    Thread.sleep(QUEUE_PROCESS_INTERVAL_MS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                } catch (Exception e) {
                    logger.warning("处理请求队列时出错: " + e.getMessage());
                }
            }
            logger.info("请求队列处理线程已结束");
        });
    }

    /**
     * 批量处理请求队列
     */
    private void processBatchRequests() throws InterruptedException {
        if (requestQueue.isEmpty()) {
            return;
        }
        
        // 检查连接状态
        if (!connected && !connect()) {
            // 如果不能连接，只处理少量请求以避免过多失败
            handleFailedRequest(3);
            return;
        }
        
        // 处理GET_ACTION请求 (优先处理)
        List<PendingRequest> actionRequests = new ArrayList<>();
        int actionCount = drainTo(requestQueue, actionRequests, BATCH_SIZE, req -> req.type == RequestType.GET_ACTION);
        
        for (PendingRequest request : actionRequests) {
            processGetActionRequest(request);
        }
        
        // 处理TRAIN请求
        List<PendingRequest> trainRequests = new ArrayList<>();
        int trainCount = drainTo(requestQueue, trainRequests, BATCH_SIZE, req -> req.type == RequestType.TRAIN);
        
        for (PendingRequest request : trainRequests) {
            processTrainRequest(request);
        }
    }

    /**
     * 从阻塞队列中按条件提取元素
     */
    private <E> int drainTo(BlockingQueue<E> source, Collection<? super E> dest, int maxElements,
            Predicate<E> filter) {
        int count = 0;
        
        while (count < maxElements) {
            E e = source.poll();
            if (e == null) {
                break; // 队列为空
            }
            
            if (filter.test(e)) {
                dest.add(e);
                count++;
            } else {
                source.offer(e); // 放回不符合条件的元素
            }
        }
        
        return count;
    }

    /**
     * 处理失败请求
     */
    private void handleFailedRequest(int maxCount) {
        int count = 0;
        while (count < maxCount && !requestQueue.isEmpty()) {
            try {
                PendingRequest request = requestQueue.poll();
                if (request != null) {
                    if (request.type == RequestType.GET_ACTION) {
                        ((Callback<double[]>) request.callback).onFailure(new IOException("未连接到Python服务器"));
                    } else {
                        ((Callback<Boolean>) request.callback).onFailure(new IOException("未连接到Python服务器"));
                    }
                    count++;
                }
            } catch (Exception e) {
                logger.warning("处理失败请求时出错: " + e.getMessage());
            }
        }
    }

    /**
     * 测试基本连接
     */
    private boolean testBasicConnection(String host, int port, int timeoutMs) {
        Socket testSocket = null;
        try {
            logger.info("测试到 " + host + ":" + port + " 的连接...");
            testSocket = new Socket();
            testSocket.connect(new InetSocketAddress(host, port), timeoutMs);
            boolean isConnected = testSocket.isConnected() && !testSocket.isClosed();
            logger.info("连接测试结果: " + (isConnected ? "成功" : "失败"));
            return isConnected;
        } catch (Exception e) {
            logger.warning("连接测试失败: " + e.getMessage());
            return false;
        } finally {
            if (testSocket != null) {
                try {
                    testSocket.close();
                } catch (IOException e) {
                    // 忽略关闭错误
                }
            }
        }
    }

    /**
     * 连接到Python服务器 - 改进版
     * @return 连接是否成功
     */
    public boolean connect() {
        boolean acquiredSemaphore = false;
        
        try {
            // 首先执行基本连接测试
            if (!testBasicConnection(host, port, 5000)) {
                logger.severe("无法建立基本网络连接，检查服务器是否运行以及网络连接是否通畅");
                return false;
            }
            
            // 使用连接信号量防止并发连接尝试
            acquiredSemaphore = connectionSemaphore.tryAcquire(5, TimeUnit.SECONDS);
            if (!acquiredSemaphore) {
                logger.warning("无法获取连接信号量，连接尝试被跳过");
                return connected; // 返回当前连接状态
            }
            
            // 检查当前连接状态
            if (socket != null && !socket.isClosed() && socket.isConnected()) {
                try {
                    // 测试连接是否真的有效
                    if (isConnectionAlive()) {
                        logger.info("当前连接仍然有效，无需重新连接");
                        return true;
                    } else {
                        logger.info("连接测试失败，将尝试重新连接");
                    }
                } catch (Exception e) {
                    logger.warning("测试连接时出错: " + e.getMessage());
                }
            }
            
            // 连接冷却检查
            long currentTime = System.currentTimeMillis();
            if (currentTime - lastConnectionAttempt < CONNECTION_COOLDOWN_MS) {
                logger.info("连接尝试过于频繁，等待冷却期结束");
                return false;
            }
            lastConnectionAttempt = currentTime;
            
            // 如果有线程在运行，停止它
            stopReading = true;
            if (responseReaderThread != null && responseReaderThread.isAlive()) {
                try {
                    logger.info("停止现有响应读取线程");
                    responseReaderThread.interrupt();
                    responseReaderThread.join(2000); // 等待2秒
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    logger.warning("等待响应读取线程时被中断");
                }
            }
            
            // 重置所有挂起的请求
            int pendingCount = pendingRequests.size();
            if (pendingCount > 0) {
                logger.info("重置 " + pendingCount + " 个挂起的请求");
                for (CompletableFuture<JSONObject> future : pendingRequests.values()) {
                    future.completeExceptionally(new IOException("连接重置"));
                }
                pendingRequests.clear();
            }
            
            // 关闭旧连接
            closeCurrentConnection();
            
            logger.info("尝试连接到Python服务器: " + host + ":" + port);
            
            try {
                // 设置socket选项 - 使用更健壮的配置
                socket = new Socket();
                socket.setReuseAddress(true);
                socket.setTcpNoDelay(true);  // 减少Nagle算法导致的延迟
                socket.setKeepAlive(true);
                socket.setSoTimeout(READ_TIMEOUT_MS);
                socket.setReceiveBufferSize(65536); // 64KB接收缓冲区
                socket.setSendBufferSize(65536); // 64KB发送缓冲区
                socket.setPerformancePreferences(0, 1, 2); // 连接时间，延迟，带宽的优先级
                
                logger.info("正在连接到 " + host + ":" + port + "，超时时间: " + CONNECTION_TIMEOUT_MS + "ms");
                socket.connect(new InetSocketAddress(host, port), CONNECTION_TIMEOUT_MS);
                logger.info("Socket连接已建立");
                
                // 创建输入输出流
                in = new BufferedReader(new InputStreamReader(socket.getInputStream(), "UTF-8"));
                out = new PrintWriter(new OutputStreamWriter(socket.getOutputStream(), "UTF-8"), true);
                
                // 更新连接状态
                connected = true;
                reconnectAttempts = 0;
                consecutiveFailures.set(0);
                stopReading = false;
                
                // 准备初始化消息
                logger.info("准备发送初始化消息");
                JSONObject initMsg = new JSONObject();
                initMsg.put("type", "init");
                initMsg.put("client_id", UUID.randomUUID().toString());
                initMsg.put("state_dimension", STATE_DIMENSION);
                initMsg.put("action_dimension", ACTION_DIMENSION);
                
                // 启动响应读取线程
                startResponseReader();
                
                // 发送初始化消息并等待响应
                logger.info("发送初始化消息并等待响应，超时时间:10秒");
                JSONObject response = sendRequestAndWaitResponse(initMsg, 10000);
                
                if (response == null) {
                    logger.severe("初始化失败，未收到服务器响应");
                    closeCurrentConnection();
                    return false;
                }
                
                if (!response.optString("status", "").equals("ok")) {
                    logger.severe("初始化失败，服务器响应状态不是OK: " + response.toString());
                    closeCurrentConnection();
                    return false;
                }
                
                logger.info("成功连接到Python服务器，响应: " + response.toString());
                
                // 启动心跳线程
                startHeartbeatThread();
                
                return true;
            } catch (IOException e) {
                connected = false;
                logger.severe("连接到Python服务器失败: " + e.getMessage());
                
                // 记录更多的诊断信息
                if (e instanceof ConnectException) {
                    logger.severe("连接被拒绝，请确认服务器正在运行且端口 " + port + " 已打开");
                } else if (e instanceof SocketTimeoutException) {
                    logger.severe("连接超时，服务器未在 " + CONNECTION_TIMEOUT_MS + "ms 内响应");
                } else if (e instanceof NoRouteToHostException) {
                    logger.severe("无法路由到主机，请检查网络配置和防火墙");
                }
                
                // 增加重连尝试次数
                reconnectAttempts++;
                if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                    long delayMs = RECONNECT_DELAY_MS * (long)Math.pow(2, reconnectAttempts - 1); // 指数退避
                    logger.info("将在 " + delayMs + "ms 后尝试重新连接 (" + reconnectAttempts + "/" + MAX_RECONNECT_ATTEMPTS + ")");
                    
                    try {
                        Thread.sleep(delayMs);
                        return connect(); // 递归重连
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        logger.warning("重连等待被中断");
                    }
                } else {
                    logger.severe("达到最大重连尝试次数 (" + MAX_RECONNECT_ATTEMPTS + ")，放弃连接");
                }
                
                return false;
            } catch (JSONException e) {
                connected = false;
                logger.severe("处理服务器响应时出错: " + e.getMessage());
                return false;
            }
        } catch (Exception e) {
            logger.severe("连接过程中发生意外错误: " + e.getMessage());
            e.printStackTrace();
            return false;
        } finally {
            // 确保在所有情况下都释放信号量
            if (acquiredSemaphore) {
                connectionSemaphore.release();
                logger.fine("连接信号量已释放");
            }
        }
    }
    
    /**
     * 测试连接是否有效
     */
    private boolean isConnectionAlive() {
        try {
            if (socket == null || socket.isClosed() || !socket.isConnected()) {
                return false;
            }
            
            // 发送心跳消息
            JSONObject heartbeat = new JSONObject();
            heartbeat.put("type", "heartbeat");
            heartbeat.put("request_id", UUID.randomUUID().toString());
            
            // 短超时快速测试连接
            JSONObject response = sendRequestAndWaitResponse(heartbeat, 3000);
            return response != null && response.optString("status", "").equals("ok");
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 关闭当前连接
     */
    private void closeCurrentConnection() {
        if (socket != null) {
            try {
                socket.close();
            } catch (IOException e) {
                logger.warning("关闭旧Socket时出错: " + e.getMessage());
            }
        }
        
        try {
            if (in != null) in.close();
        } catch (IOException e) {}
        
        if (out != null) out.close();
        
        socket = null;
        in = null;
        out = null;
        connected = false;
    }

    /**
     * 断开与Python服务器的连接 - 改进版
     */
    public synchronized void disconnect() {
        try {
            stopReading = true; // 停止响应读取线程
            processingQueue = false; // 停止队列处理
            
            if (socket != null && !socket.isClosed()) {
                logger.info("断开与Python服务器的连接");
                
                // 发送关闭消息
                try {
                    JSONObject closeMsg = new JSONObject();
                    closeMsg.put("type", "close");
                    sendMessage(closeMsg);
                    
                    // 等待一段时间确保消息发送
                    try {
                        Thread.sleep(200);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                } catch (Exception e) {
                    logger.warning("发送关闭消息失败: " + e.getMessage());
                }
                
                // 安全关闭Socket
                try {
                    if (socket.getInputStream() != null) socket.getInputStream().close();
                    if (socket.getOutputStream() != null) socket.getOutputStream().close();
                    socket.close();
                    logger.info("Socket已安全关闭");
                } catch (IOException e) {
                    logger.warning("关闭Socket时出错: " + e.getMessage());
                }
                
                socket = null;
                in = null;
                out = null;
                connected = false;
            }
            
            // 完成所有挂起的请求
            for (CompletableFuture<JSONObject> future : pendingRequests.values()) {
                future.completeExceptionally(new IOException("连接已关闭"));
            }
            pendingRequests.clear();
            
            // 完成所有队列中的请求
            List<PendingRequest> remainingRequests = new ArrayList<>();
            requestQueue.drainTo(remainingRequests);
            for (PendingRequest req : remainingRequests) {
                if (req.type == RequestType.GET_ACTION) {
                    ((Callback<double[]>) req.callback).onFailure(new IOException("连接已关闭"));
                } else {
                    ((Callback<Boolean>) req.callback).onFailure(new IOException("连接已关闭"));
                }
            }
            
        } catch (Exception e) {
            logger.severe("断开连接时出错: " + e.getMessage());
        }
    }

    /**
     * 改进的读取完整响应方法 - 直接使用字节流
     */
    private String readCompleteResponse(BufferedReader reader) throws IOException {
        try {
            // 直接使用原始输入流读取字节，绕过BufferedReader的行缓冲
            InputStream inputStream = socket.getInputStream();
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            byte[] data = new byte[READ_BUFFER_SIZE];
            int bytesRead;
            
            // 设置超时
            long startTime = System.currentTimeMillis();
            
            while (System.currentTimeMillis() - startTime < READ_TIMEOUT_MS) {
                // 检查是否有可用数据
                if (inputStream.available() > 0) {
                    bytesRead = inputStream.read(data);
                    if (bytesRead > 0) {
                        logger.info("已读取 " + bytesRead + " 字节的数据");
                        buffer.write(data, 0, bytesRead);
                        
                        // 检查是否有完整消息
                        String bufferStr = buffer.toString("UTF-8");
                        
                        // 检查消息边界
                        int boundaryIndex = bufferStr.indexOf(MESSAGE_BOUNDARY);
                        if (boundaryIndex >= 0) {
                            String message = bufferStr.substring(0, boundaryIndex);
                            logger.info("接收到完整消息 (消息边界): " + message);
                            return message;
                        }
                        
                        // 尝试检查是否有有效JSON
                        try {
                            new JSONObject(bufferStr);
                            // 是完整JSON
                            logger.info("接收到完整JSON: " + bufferStr);
                            return bufferStr;
                        } catch (JSONException e) {
                            // 不是完整JSON，继续接收
                        }
                    }
                } else {
                    // 当前没有数据，短暂休眠
                    Thread.sleep(50);
                    
                    // 如果已经有数据但一段时间没有新数据，可能是消息已完整但没有边界
                    if (buffer.size() > 0 && (System.currentTimeMillis() - startTime > READ_TIMEOUT_MS / 2)) {
                        String bufferStr = buffer.toString("UTF-8");
                        try {
                            new JSONObject(bufferStr);
                            // 是完整JSON
                            logger.info("接收到完整JSON (超时后检查): " + bufferStr);
                            return bufferStr;
                        } catch (JSONException e) {
                            // 不是完整JSON，继续等待
                        }
                    }
                }
            }
            
            // 超时，但有部分数据
            if (buffer.size() > 0) {
                String bufferStr = buffer.toString("UTF-8");
                logger.warning("读取超时但有部分数据: " + bufferStr);
                
                // 尝试提取JSON部分
                int jsonStart = bufferStr.indexOf('{');
                int jsonEnd = bufferStr.lastIndexOf('}');
                if (jsonStart >= 0 && jsonEnd > jsonStart) {
                    String jsonPart = bufferStr.substring(jsonStart, jsonEnd + 1);
                    try {
                        new JSONObject(jsonPart);
                        logger.info("从部分数据中提取到JSON: " + jsonPart);
                        return jsonPart;
                    } catch (JSONException e) {
                        // 无效JSON
                    }
                }
                
                // 返回原始数据
                return bufferStr;
            }
            
            // 超时且无数据
            logger.warning("读取超时且未收到数据");
            return "";
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("读取被中断", e);
        } catch (Exception e) {
            logger.severe("读取响应时出错: " + e.getMessage());
            throw new IOException("读取响应失败", e);
        }
    }

    /**
     * 启动响应读取线程
     */
    private void startResponseReader() {
        responseReaderThread = new Thread(() -> {
            logger.info("启动响应读取线程");
            try {
                while (!stopReading && connected && !Thread.currentThread().isInterrupted()) {
                    try {
                        // 检查socket是否连接
                        if (socket == null || !socket.isConnected() || socket.isClosed()) {
                            logger.warning("Socket已关闭或未连接，响应读取线程退出");
                            connected = false;
                            break;
                        }
                        
                        // 使用改进的响应读取方法
                        String responseStr = readCompleteResponse(in);
                        
                        if (responseStr == null || responseStr.isEmpty()) {
                            // 没有数据，短暂休眠后继续
                            Thread.sleep(50);
                            continue;
                        }
                        
                        // 记录收到的完整响应
                        logger.info("收到完整响应: " + responseStr);
                        
                        // 处理响应
                        handleResponse(responseStr);
                        consecutiveFailures.set(0); // 重置失败计数器
                        
                    } catch (SocketTimeoutException e) {
                        logger.warning("读取响应超时: " + e.getMessage());
                        // 继续尝试读取，不退出循环
                    } catch (IOException e) {
                        if (stopReading) {
                            // 正常停止
                            break;
                        }
                        logger.warning("读取响应时发生IO错误: " + e.getMessage());
                        
                        // 增加失败计数
                        int failures = consecutiveFailures.incrementAndGet();
                        logger.info("连续失败次数: " + failures);
                        
                        if (failures > MAX_CONSECUTIVE_FAILURES) {
                            logger.severe("连续失败次数超过阈值，强制重连");
                            connected = false;
                            break;
                        }
                        
                        // 短暂休眠后继续尝试读取
                        try {
                            Thread.sleep(100);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                        }
                    } catch (InterruptedException e) {
                        logger.info("响应读取线程被中断");
                        Thread.currentThread().interrupt();
                        break;
                    } catch (Exception e) {
                        logger.warning("处理响应时出错: " + e.getMessage());
                        e.printStackTrace();
                        
                        // 非致命错误处理，短暂休眠后继续
                        try {
                            Thread.sleep(100);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            break;
                        }
                    }
                }
            } catch (Exception e) {
                logger.severe("响应读取线程异常: " + e.getMessage());
                e.printStackTrace();
            } finally {
                logger.info("响应读取线程结束");
                
                // 如果连接仍应该处于活动状态但线程已结束，则重启
                if (connected && !stopReading) {
                    executor.submit(() -> {
                        try {
                            logger.info("准备重启响应读取线程");
                            Thread.sleep(1000); // 等待1秒
                            if (connected && !stopReading) {
                                startResponseReader(); // 重启读取线程
                            }
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    });
                }
            }
        });
        
        responseReaderThread.setName("ResponseReader");
        responseReaderThread.setDaemon(true);
        responseReaderThread.start();
    }

    /**
     * 处理接收到的响应
     * @param responseStr 响应字符串
     */
    private void handleResponse(String responseStr) {
        if (responseStr == null || responseStr.isEmpty()) {
            logger.warning("收到空响应");
            return;
        }
        
        try {
            // 尝试清理响应字符串 - 查找JSON内容
            String jsonStr = responseStr.trim();
            
            // 如果响应不是以{开始，尝试找到JSON开始位置
            if (!jsonStr.startsWith("{")) {
                int jsonStart = jsonStr.indexOf('{');
                if (jsonStart >= 0) {
                    jsonStr = jsonStr.substring(jsonStart);
                    logger.info("提取JSON部分，从位置" + jsonStart + "开始: " + jsonStr);
                }
            }
            
            // 如果响应不是以}结束，尝试找到JSON结束位置
            if (!jsonStr.endsWith("}")) {
                int jsonEnd = jsonStr.lastIndexOf('}');
                if (jsonEnd >= 0) {
                    jsonStr = jsonStr.substring(0, jsonEnd + 1);
                    logger.info("提取JSON部分，到位置" + jsonEnd + "结束: " + jsonStr);
                }
            }
            
            // 解析响应JSON
            JSONObject response = new JSONObject(jsonStr);
            String requestId = response.optString("request_id", "");
            
            logger.info("成功解析响应JSON，请求ID: " + requestId);
            
            // 检查是否有挂起的请求对应此响应
            CompletableFuture<JSONObject> future = pendingRequests.remove(requestId);
            if (future != null) {
                logger.info("找到对应的挂起请求，完成Future");
                future.complete(response);
            } else {
                // 可能是心跳响应或服务器主动发送的消息
                logger.info("收到未关联请求ID的响应: " + jsonStr);
            }
        } catch (JSONException e) {
            logger.warning("解析响应JSON时出错: " + e.getMessage() + ", 响应: " + responseStr);
            e.printStackTrace();
            
            // 尝试从响应中提取有效的JSON部分
            try {
                String jsonPart = extractJsonPart(responseStr);
                if (jsonPart != null) {
                    logger.info("尝试提取的JSON部分: " + jsonPart);
                    JSONObject response = new JSONObject(jsonPart);
                    String requestId = response.optString("request_id", "");
                    
                    CompletableFuture<JSONObject> future = pendingRequests.remove(requestId);
                    if (future != null) {
                        logger.info("使用提取的JSON部分完成Future");
                        future.complete(response);
                    }
                }
            } catch (Exception ex) {
                logger.warning("尝试提取JSON部分时出错: " + ex.getMessage());
            }
        }
    }

    /**
     * 从响应字符串中提取JSON部分
     * @param responseStr 响应字符串
     * @return 提取的JSON字符串，如果无法提取则返回null
     */
    private String extractJsonPart(String responseStr) {
        try {
            int startBrace = responseStr.indexOf('{');
            int endBrace = responseStr.lastIndexOf('}');
            
            if (startBrace >= 0 && endBrace > startBrace) {
                // 尝试提取JSON部分
                String jsonPart = responseStr.substring(startBrace, endBrace + 1);
                // 验证是否为有效JSON
                new JSONObject(jsonPart);
                return jsonPart;
            }
        } catch (Exception e) {
            // 提取或验证JSON失败
        }
        return null;
    }

    /**
     * 发送请求并等待响应，同步方法 - 改进版
     * @param request 请求对象
     * @param timeoutMs 超时时间(毫秒)
     * @return 响应对象，超时或错误则返回null
     */
    private JSONObject sendRequestAndWaitResponse(JSONObject request, int timeoutMs) throws IOException, JSONException {
        if (!connected && !connect()) {
            throw new IOException("未连接到Python服务器");
        }
        
        // 如果没有请求ID则添加
        if (!request.has("request_id")) {
            request.put("request_id", UUID.randomUUID().toString());
        }
        String requestId = request.getString("request_id");
        
        // 创建Future用于等待响应
        CompletableFuture<JSONObject> responseFuture = new CompletableFuture<>();
        pendingRequests.put(requestId, responseFuture);
        
        try {
            // 发送请求
            logger.info("发送请求: " + request.toString());
            boolean sendSuccess = sendMessage(request);
            if (!sendSuccess) {
                pendingRequests.remove(requestId);
                throw new IOException("发送请求失败");
            }
            
            // 等待响应或超时
            return responseFuture.get(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("等待响应时中断", e);
        } catch (ExecutionException e) {
            throw new IOException("获取响应出错", e.getCause());
        } catch (TimeoutException e) {
            logger.severe("等待响应ID " + requestId + " 超时，检查是否收到部分响应");
            pendingRequests.remove(requestId);
            throw new IOException("等待响应超时", e);
        }
    }

    /**
     * 异步发送状态并获取动作
     * @param state 状态数据
     * @param callback 回调函数，用于处理返回的动作
     * @param algorithm 要使用的算法，可以是null(使用默认算法)
     */
    public void getActionAsync(double[] state, Callback<double[]> callback, String algorithm) {
        // 添加到请求队列
        PendingRequest request = new PendingRequest(RequestType.GET_ACTION, state, null, 0.0, null, false, callback, algorithm);
        requestQueue.offer(request);
    }

    /**
     * 异步发送状态并获取动作（使用默认算法）
     * @param state 状态数据
     * @param callback 回调函数，用于处理返回的动作
     */
    public void getActionAsync(double[] state, Callback<double[]> callback) {
        getActionAsync(state, callback, null);
    }

    /**
     * 异步训练RL代理
     * @param state 当前状态
     * @param action 执行的动作
     * @param reward 获得的奖励
     * @param nextState 下一个状态
     * @param done 是否结束
     * @param callback 回调函数
     * @param algorithm 要使用的算法，可以是null(使用默认算法)
     */
    public void trainAsync(double[] state, double[] action, double reward, 
                        double[] nextState, boolean done, Callback<Boolean> callback, String algorithm) {
        // 添加到请求队列
        PendingRequest request = new PendingRequest(RequestType.TRAIN, state, action, reward, nextState, done, callback, algorithm);
        requestQueue.offer(request);
    }

    /**
     * 异步训练RL代理（使用默认算法）
     * @param state 当前状态
     * @param action 执行的动作
     * @param reward 获得的奖励
     * @param nextState 下一个状态
     * @param done 是否结束
     * @param callback 回调函数
     */
    public void trainAsync(double[] state, double[] action, double reward, 
                        double[] nextState, boolean done, Callback<Boolean> callback) {
        trainAsync(state, action, reward, nextState, done, callback, null);
    }

    /**
     * 处理获取动作请求
     */
    private void processGetActionRequest(PendingRequest request) {
        executor.submit(() -> {
            try {
                if (!connected && !connect()) {
                    ((Callback<double[]>) request.callback).onFailure(new IOException("未连接到Python服务器"));
                    return;
                }
                
                JSONObject jsonRequest = new JSONObject();
                jsonRequest.put("type", "get_action");
                // 添加请求ID
                String requestId = UUID.randomUUID().toString();
                jsonRequest.put("request_id", requestId);
                
                // 标准化状态数据
                double[] standardizedState = standardizeState(request.state);
                JSONArray stateArray = new JSONArray();
                for (double value : standardizedState) {
                    stateArray.put(value);
                }
                jsonRequest.put("state", stateArray);
                
                // 添加算法参数
                String algorithm = request.algorithmName != null ? request.algorithmName : defaultAlgorithm;
                jsonRequest.put("algorithm", algorithm);
                logger.info("使用算法 " + algorithm + " 处理获取动作请求");
                
                // 创建Future用于等待响应
                CompletableFuture<JSONObject> responseFuture = new CompletableFuture<>();
                pendingRequests.put(requestId, responseFuture);
                
                // 设置超时处理
                ScheduledFuture<?> timeoutFuture = requestTimeoutScheduler.schedule(() -> {
                    if (!responseFuture.isDone()) {
                        responseFuture.completeExceptionally(new TimeoutException("请求超时"));
                        pendingRequests.remove(requestId);
                    }
                }, READ_TIMEOUT_MS, TimeUnit.MILLISECONDS);
                
                long startTime = System.currentTimeMillis();
                
                // 发送请求
                boolean sendSuccess = sendMessage(jsonRequest);
                if (!sendSuccess) {
                    timeoutFuture.cancel(false);
                    pendingRequests.remove(requestId);
                    ((Callback<double[]>) request.callback).onFailure(new IOException("发送请求失败"));
                    return;
                }
                
                // 异步处理响应
                responseFuture.whenComplete((response, exception) -> {
                    timeoutFuture.cancel(false);
                    long endTime = System.currentTimeMillis();
                    
                    // 更新统计信息
                    synchronized (lock) {
                        updateStatistics("action_request_count", 1, true);
                        updateStatistics("action_request_time_ms", endTime - startTime, false);
                    }
                    
                    if (exception != null) {
                        logger.warning("获取动作时出错: " + exception.getMessage());
                        ((Callback<double[]>) request.callback).onFailure(new IOException("获取动作失败", exception));
                        // 如果是连接问题，尝试重连
                        handleConnectionException(exception);
                        return;
                    }
                    
                    try {
                        if (response != null && response.has("action")) {
                            JSONArray actionArray = response.getJSONArray("action");
                            double[] action = new double[actionArray.length()];
                            for (int i = 0; i < actionArray.length(); i++) {
                                action[i] = actionArray.getDouble(i);
                            }
                            ((Callback<double[]>) request.callback).onSuccess(action);
                        } else {
                            String errorMsg = (response != null && response.has("error")) 
                                ? response.getString("error") : "响应中没有动作数据";
                            ((Callback<double[]>) request.callback).onFailure(new IOException(errorMsg));
                        }
                    } catch (Exception e) {
                        ((Callback<double[]>) request.callback).onFailure(e);
                    }
                });
            } catch (Exception e) {
                logger.warning("创建动作请求时出错: " + e.getMessage());
                ((Callback<double[]>) request.callback).onFailure(e);
            }
        });
    }

    /**
     * 处理训练请求
     */
    private void processTrainRequest(PendingRequest request) {
        executor.submit(() -> {
            try {
                if (!connected && !connect()) {
                    ((Callback<Boolean>) request.callback).onFailure(new IOException("未连接到Python服务器"));
                    return;
                }
                
                JSONObject jsonRequest = new JSONObject();
                jsonRequest.put("type", "update");  // 使用update类型匹配服务器期望
                
                // 添加请求ID
                String requestId = UUID.randomUUID().toString();
                jsonRequest.put("request_id", requestId);
                
                // 标准化状态和动作维度
                double[] standardizedState = standardizeState(request.state);
                double[] standardizedAction = standardizeAction(request.action);
                double[] standardizedNextState = standardizeState(request.nextState);
                
                // 构建JSON数组
                JSONArray stateArray = new JSONArray();
                for (double value : standardizedState) {
                    stateArray.put(value);
                }
                jsonRequest.put("state", stateArray);
                
                JSONArray actionArray = new JSONArray();
                for (double value : standardizedAction) {
                    actionArray.put(value);
                }
                jsonRequest.put("action", actionArray);
                
                jsonRequest.put("reward", request.reward);
                
                JSONArray nextStateArray = new JSONArray();
                for (double value : standardizedNextState) {
                    nextStateArray.put(value);
                }
                jsonRequest.put("next_state", nextStateArray);
                
                jsonRequest.put("done", request.done);
                
                // 添加算法参数
                String algorithm = request.algorithmName != null ? request.algorithmName : defaultAlgorithm;
                jsonRequest.put("algorithm", algorithm);
                logger.info("使用算法 " + algorithm + " 处理训练请求");
                
                // 创建Future用于等待响应
                CompletableFuture<JSONObject> responseFuture = new CompletableFuture<>();
                pendingRequests.put(requestId, responseFuture);
                
                // 设置超时处理
                ScheduledFuture<?> timeoutFuture = requestTimeoutScheduler.schedule(() -> {
                    if (!responseFuture.isDone()) {
                        responseFuture.completeExceptionally(new TimeoutException("训练请求超时"));
                        pendingRequests.remove(requestId);
                    }
                }, READ_TIMEOUT_MS, TimeUnit.MILLISECONDS);
                
                long startTime = System.currentTimeMillis();
                
                // 发送请求
                boolean sendSuccess = sendMessage(jsonRequest);
                if (!sendSuccess) {
                    timeoutFuture.cancel(false);
                    pendingRequests.remove(requestId);
                    ((Callback<Boolean>) request.callback).onFailure(new IOException("发送请求失败"));
                    return;
                }
                
                logger.info("已发送训练请求: " + requestId);
                
                // 异步处理响应
                responseFuture.whenComplete((response, exception) -> {
                    timeoutFuture.cancel(false);
                    long endTime = System.currentTimeMillis();
                    
                    // 更新统计信息
                    synchronized (lock) {
                        updateStatistics("train_request_count", 1, true);
                        updateStatistics("train_request_time_ms", endTime - startTime, false);
                    }
                    
                    if (exception != null) {
                        logger.warning("训练时出错: " + exception.getMessage());
                        ((Callback<Boolean>) request.callback).onFailure(new IOException("训练失败", exception));
                        
                        // 如果是连接问题，尝试重连
                        handleConnectionException(exception);
                        return;
                    }
                    
                    try {
                        if (response != null && 
                            (response.has("status") && response.getString("status").equals("success")) || 
                            (response.has("success") && response.getBoolean("success"))) {
                            logger.info("训练请求成功: " + requestId);
                            ((Callback<Boolean>) request.callback).onSuccess(true);
                        } else if (response != null && response.has("error")) {
                            String errorMsg = response.getString("error");
                            logger.warning("训练请求失败: " + errorMsg);
                            ((Callback<Boolean>) request.callback).onFailure(new IOException("训练失败: " + errorMsg));
                        } else {
                            logger.warning("训练响应没有成功状态: " + (response != null ? response.toString() : "null"));
                            ((Callback<Boolean>) request.callback).onFailure(new IOException("响应中没有成功状态"));
                        }
                    } catch (Exception e) {
                        logger.warning("处理训练响应时出错: " + e.getMessage());
                        ((Callback<Boolean>) request.callback).onFailure(e);
                    }
                });
            } catch (Exception e) {
                logger.warning("创建训练请求时出错: " + e.getMessage());
                ((Callback<Boolean>) request.callback).onFailure(e);
            }
        });
    }

    /**
     * 处理连接异常
     */
    private void handleConnectionException(Throwable exception) {
        if (exception instanceof IOException || exception.getCause() instanceof IOException) {
            logger.warning("检测到连接问题，将尝试使用退避策略重新连接");
            connected = false;
            reconnectWithExponentialBackoff();
        }
    }

    /**
     * 标准化状态维度到12维
     * @param state 原始状态数组
     * @return 标准化后的12维状态数组
     */
    private double[] standardizeState(double[] state) {
        // 始终返回12维状态向量
        double[] standardized = new double[STATE_DIMENSION];
        
        // 复制原始数组中的值，不足补0，超出截断
        for (int i = 0; i < STATE_DIMENSION; i++) {
            if (i < state.length) {
                standardized[i] = state[i];
            } else {
                standardized[i] =.0;
            }
        }
        
        return standardized;
    }
    
    /**
     * 标准化动作维度到3维
     * @param action 原始动作数组
     * @return 标准化后的3维动作数组
     */
    private double[] standardizeAction(double[] action) {
        // 始终返回3维动作向量
        double[] standardized = new double[ACTION_DIMENSION];
        
        // 复制原始数组中的值，不足补0，超出截断
        for (int i = 0; i < ACTION_DIMENSION; i++) {
            if (i < action.length) {
                standardized[i] = action[i];
            } else {
                standardized[i] = 0.0;
            }
        }
        
        return standardized;
    }

    /**
     * 发送消息到Python服务器，修复发送格式以匹配Python服务器期望
     * @param message JSON消息
     * @return 是否发送成功
     */
    private synchronized boolean sendMessage(JSONObject message) {
        try {
            if (socket == null || !socket.isConnected() || socket.isClosed()) {
                logger.warning("Socket未连接或已关闭，无法发送消息");
                return false;
            }
            
            String msgStr = message.toString();
            // 直接发送消息，添加双换行符作为消息边界
            byte[] msgBytes = (msgStr + MESSAGE_BOUNDARY).getBytes(StandardCharsets.UTF_8);
            
            // 使用底层输出流直接发送
            socket.getOutputStream().write(msgBytes);
            socket.getOutputStream().flush();
            
            logger.info("已发送消息: " + msgStr);
            return true;
        } catch (Exception e) {
            logger.warning("发送消息时出错: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * 指数退避策略的重连
     */
    private void reconnectWithExponentialBackoff() {
        executor.submit(() -> {
            try {
                // 指数退避重连
                int attemptCount = 0;
                int maxAttempts = 5;
                boolean reconnected = false;
                
                while (!reconnected && attemptCount < maxAttempts) {
                    long delay = (long) (1000 * Math.pow(2, attemptCount));
                    logger.info("等待 " + delay + "ms 后尝试重新连接...");
                    Thread.sleep(delay);
                    
                    reconnected = connect();
                    if (reconnected) {
                        logger.info("重连成功");
                        break;
                    }
                    
                    attemptCount++;
                    logger.warning("重连尝试 " + attemptCount + "/" + maxAttempts + " 失败");
                }
                
                if (!reconnected) {
                    logger.severe("重连失败，达到最大尝试次数");
                }
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
        });
    }

    /**
     * 启动心跳线程，保持连接活跃
     */
    private void startHeartbeatThread() {
        executor.submit(() -> {
            try {
                while (connected && !Thread.currentThread().isInterrupted()) {
                    try {
                        Thread.sleep(15000);  // 每15秒发送一次心跳
                        
                        if (!connected) {
                            break;
                        }
                        
                        JSONObject heartbeat = new JSONObject();
                        heartbeat.put("type", "heartbeat");
                        heartbeat.put("request_id", UUID.randomUUID().toString());
                        
                        // 创建Future用于等待响应
                        CompletableFuture<JSONObject> responseFuture = new CompletableFuture<>();
                        pendingRequests.put(heartbeat.getString("request_id"), responseFuture);
                        
                        // 设置超时处理
                        ScheduledFuture<?> timeoutFuture = requestTimeoutScheduler.schedule(() -> {
                            if (!responseFuture.isDone()) {
                                responseFuture.completeExceptionally(new TimeoutException("心跳请求超时"));
                                pendingRequests.remove(heartbeat.getString("request_id"));
                            }
                        }, 5000, TimeUnit.MILLISECONDS);
                        
                        // 发送心跳请求
                        boolean sendSuccess = sendMessage(heartbeat);
                        if (!sendSuccess) {
                            timeoutFuture.cancel(false);
                            pendingRequests.remove(heartbeat.getString("request_id"));
                            logger.warning("发送心跳请求失败");
                            connected = false;
                            // 尝试重新连接
                            executor.submit(() -> connect());
                            break;
                        }
                        
                        // 异步处理响应
                        responseFuture.whenComplete((response, exception) -> {
                            timeoutFuture.cancel(false);
                            
                            if (exception != null) {
                                logger.warning("心跳检测失败: " + exception.getMessage());
                                connected = false;
                                // 尝试重新连接
                                executor.submit(() -> connect());
                            } else if (!response.has("status") || !response.getString("status").equals("ok")) {
                                logger.warning("心跳检测失败，服务器响应异常: " + response.toString());
                                connected = false;
                                // 尝试重新连接
                                executor.submit(() -> connect());
                            }
                        });
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    } catch (Exception e) {
                        logger.warning("心跳检测时出错: " + e.getMessage());
                        
                        // 如果连接断开，尝试重新连接
                        if (!connected || e instanceof IOException) {
                            connected = false;
                            reconnectWithExponentialBackoff();
                            break;
                        }
                    }
                }
            } catch (Exception e) {
                logger.severe("心跳线程异常终止: " + e.getMessage());
            } finally {
                logger.info("心跳线程已终止");
            }
        });
    }

    /**
     * 发送心跳并等待响应
     */
    private boolean sendHeartbeat(JSONObject heartbeat) {
        try {
            // 创建Future用于等待响应
            String requestId = heartbeat.getString("request_id");
            CompletableFuture<JSONObject> responseFuture = new CompletableFuture<>();
            pendingRequests.put(requestId, responseFuture);
            
            // 发送心跳请求
            boolean sendSuccess = sendMessage(heartbeat);
            if (!sendSuccess) {
                pendingRequests.remove(requestId);
                return false;
            }
            
            // 等待响应最多5秒
            JSONObject response = responseFuture.get(5000, TimeUnit.MILLISECONDS);
            
            // 验证响应
            return response != null && 
                   (response.has("status") && "ok".equals(response.getString("status")));
                
        } catch (Exception e) {
            logger.warning("心跳检测失败: " + e.getMessage());
            return false;
        }
    }

    /**
     * 获取连接状态
     * @return 是否已连接
     */
    public boolean isConnected() {
        return connected && socket != null && !socket.isClosed() && socket.isConnected();
    }

    /**
     * 获取统计信息
     * @return 统计信息Map
     */
    public Map<String, Object> getStatistics() {
        return new HashMap<>(statistics);
    }

    /**
     * 更新统计信息
     * @param key 统计项键
     * @param value 值
     * @param increment 是否增量更新
     */
    private void updateStatistics(String key, Object value, boolean increment) {
        if (increment && statistics.containsKey(key) && statistics.get(key) instanceof Number) {
            Number currentValue = (Number) statistics.get(key);
            if (value instanceof Integer) {
                statistics.put(key, currentValue.intValue() + ((Integer) value));
            } else if (value instanceof Long) {
                statistics.put(key, currentValue.longValue() + ((Long) value));
            } else if (value instanceof Double) {
                statistics.put(key, currentValue.doubleValue() + ((Double) value));
            }
        } else {
            statistics.put(key, value);
        }
        
        // 如果是时间统计，计算平均值
        if (key.endsWith("_time_ms") && increment == false) {
            String countKey = key.replace("_time_ms", "_count");
            if (statistics.containsKey(countKey)) {
                int count = (Integer) statistics.get(countKey);
                String avgKey = key.replace("_time_ms", "_avg_time_ms");
                
                if (statistics.containsKey(avgKey)) {
                    double currentAvg = (Double) statistics.get(avgKey);
                    double newValue = (double) value;
                    statistics.put(avgKey, (currentAvg * (count - 1) + newValue) / count);
                } else {
                    statistics.put(avgKey, (double) value);
                }
            }
        }
    }

    /**
     * 关闭所有资源
     */
    public void shutdown() {
        try {
            disconnect();
            
            // 停止请求处理
            processingQueue = false;
            
            // 关闭线程池
            executor.shutdown();
            try {
                if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
            
            // 关闭超时调度器
            requestTimeoutScheduler.shutdown();
            try {
                if (!requestTimeoutScheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                    requestTimeoutScheduler.shutdownNow();
                }
            } catch (InterruptedException e) {
                requestTimeoutScheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
            
            logger.info("接口已完全关闭");
        } catch (Exception e) {
            logger.severe("关闭资源时出错: " + e.getMessage());
        }
    }
    
    /**
     * 内部类，表示请求类型
     */
    private enum RequestType {
        GET_ACTION,
        TRAIN
    }
    
    /**
     * 内部接口，作为Java 8函数式接口的替代
     */
    public interface Predicate<T> {
        boolean test(T t);
    }
    
    /**
     * 内部类，表示等待处理的请求
     */
    private class PendingRequest {
        final RequestType type;
        final double[] state;
        final double[] action;
        final double reward;
        final double[] nextState;
        final boolean done;
        final Object callback;
        final String algorithmName;
        
        public PendingRequest(RequestType type, double[] state, double[] action, 
                             double reward, double[] nextState, boolean done, 
                             Object callback, String algorithmName) {
            this.type = type;
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.done = done;
            this.callback = callback;
            this.algorithmName = algorithmName;
        }
    }
    
    /**
     * 回调接口
     */
    public interface Callback<T> {
        void onSuccess(T result);
        void onFailure(Exception e);
    }
}