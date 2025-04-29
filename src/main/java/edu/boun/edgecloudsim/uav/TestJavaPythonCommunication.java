package edu.boun.edgecloudsim.uav;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;
import java.util.logging.Level;
import java.util.logging.FileHandler;
import java.util.logging.SimpleFormatter;


/**
 * 测试Java和Python之间的通信
 * 用于测试增强型Python接口的功能和性能
 */
public class TestJavaPythonCommunication {
    // 测试配置
    private static final int DEFAULT_STATE_DIM = 12;
    private static final int DEFAULT_ACTION_DIM = 3;
    private static final Random random = new Random();
    private static Logger logger;
    
    // 算法列表 - 用于测试不同的强化学习算法
    private static final String[] ALGORITHMS = {
        EnhancedPythonInterface.ALGORITHM_PRIORITIZED_DDPG,
        EnhancedPythonInterface.ALGORITHM_DDPG,
        EnhancedPythonInterface.ALGORITHM_DQN,
        EnhancedPythonInterface.ALGORITHM_PPO
    };

    static {
        // 配置日志
        try {
            logger = Logger.getLogger(TestJavaPythonCommunication.class.getName());
            FileHandler fileHandler = new FileHandler("test_java_python_communication.log");
            fileHandler.setFormatter(new SimpleFormatter());
            logger.addHandler(fileHandler);
            logger.setLevel(Level.INFO);
        } catch (Exception e) {
            System.err.println("无法设置日志记录器: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        System.out.println("开始Java-Python通信测试...");
        
        // 创建Python接口
        EnhancedPythonInterface pythonInterface = new EnhancedPythonInterface("localhost", 12345);
        
        try {
            // 连接到Python服务器
            System.out.println("正在连接到Python服务器...");
            boolean connected = pythonInterface.connect();
            
            if (!connected) {
                System.err.println("无法连接到Python服务器，测试中止");
                return;
            }
            
            System.out.println("成功连接到Python服务器");
            
            // 设置测试参数
            int numRequests = 50;        // 请求总数
            int concurrentRequests = 10; // 并发请求数
            
            // 运行完整测试序列
            try {
                // 1. 测试获取动作
                testGetAction(pythonInterface, numRequests, concurrentRequests);
                
                // 等待完成
                Thread.sleep(1000);
                
                // 2. 测试训练
                testTrain(pythonInterface, numRequests, concurrentRequests);
                
                // 等待完成
                Thread.sleep(1000);
                
                // 3. 测试故障容错
                testFaultTolerance(pythonInterface);
                
                // 等待完成
                Thread.sleep(1000);
                
                // 4. 测试性能
                testPerformance(pythonInterface, numRequests, concurrentRequests);
                
                // 等待完成
                Thread.sleep(1000);
                
                // 5. 快速请求压力测试
                testStressWithRapidRequests(pythonInterface, numRequests);
                
                // 打印统计信息
                System.out.println("\n===== 统计信息 =====");
                pythonInterface.getStatistics().forEach((key, value) -> {
                    System.out.println(key + ": " + value);
                });
                
            } catch (Exception e) {
                System.err.println("测试过程中出错: " + e.getMessage());
                e.printStackTrace();
            }
            
            // 断开连接
            System.out.println("\n正在断开连接...");
            pythonInterface.disconnect();
            
        } finally {
            // 关闭资源
            System.out.println("关闭资源...");
            pythonInterface.shutdown();
            System.out.println("测试完成");
        }
    }
    
    /**
     * 测试获取动作功能 - 使用多种算法
     */
    private static void testGetAction(EnhancedPythonInterface pythonInterface, int numRequests, int concurrentRequests) 
            throws InterruptedException {
        System.out.println("\n===== 测试获取动作 =====");
        
        // 创建倒计时锁
        CountDownLatch latch = new CountDownLatch(numRequests);
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger activeRequests = new AtomicInteger(0);
        
        long startTime = System.currentTimeMillis();
        
        // 发送多个请求
        for (int i = 0; i < numRequests; i++) {
            final int requestId = i;
            
            // 控制并发请求数
            while (activeRequests.get() >= concurrentRequests) {
                Thread.sleep(10);
            }
            
            activeRequests.incrementAndGet();
            
            // 创建随机状态
            double[] state = generateRandomState(DEFAULT_STATE_DIM);
            
            // 选择算法，轮流使用不同算法进行测试
            String algorithm = ALGORITHMS[i % ALGORITHMS.length];
            System.out.println("请求 " + requestId + " 使用算法: " + algorithm);
            
            pythonInterface.getActionAsync(state, new EnhancedPythonInterface.Callback<double[]>() {
                @Override
                public void onSuccess(double[] action) {
                    System.out.println("请求 " + requestId + " 成功，收到动作: " + formatArray(action));
                    successCount.incrementAndGet();
                    activeRequests.decrementAndGet();
                    latch.countDown();
                }
                
                @Override
                public void onFailure(Exception e) {
                    System.out.println("请求 " + requestId + " 失败: " + e.getMessage());
                    activeRequests.decrementAndGet();
                    latch.countDown();
                }
            }, algorithm);
            
            // 小延迟以避免服务器过载
            if (i % 10 == 0) {
                Thread.sleep(50);
            }
        }
        
        // 等待所有请求完成
        if (!latch.await(numRequests * 500, TimeUnit.MILLISECONDS)) {
            System.out.println("警告: 部分请求未在预期时间内完成");
        }
        
        long endTime = System.currentTimeMillis();
        double elapsedSec = (endTime - startTime) / 1000.0;
        
        System.out.println("\n获取动作测试结果:");
        System.out.println("成功请求: " + successCount.get() + "/" + numRequests);
        System.out.println("总耗时: " + elapsedSec + " 秒");
        System.out.println("平均每秒处理: " + (numRequests / elapsedSec) + " 请求");
    }
    
    /**
     * 测试训练功能 - 使用多种算法
     */
    private static void testTrain(EnhancedPythonInterface pythonInterface, int numRequests, int concurrentRequests) 
            throws InterruptedException {
        System.out.println("\n===== 测试训练 =====");
        
        // 创建倒计时锁
        CountDownLatch latch = new CountDownLatch(numRequests);
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger activeRequests = new AtomicInteger(0);
        
        long startTime = System.currentTimeMillis();
        
        // 发送多个请求
        for (int i = 0; i < numRequests; i++) {
            final int requestId = i;
            
            // 控制并发请求数
            while (activeRequests.get() >= concurrentRequests) {
                Thread.sleep(10);
            }
            
            activeRequests.incrementAndGet();
            
            // 创建随机训练数据
            double[] state = generateRandomState(DEFAULT_STATE_DIM);
            double[] action = generateRandomAction(DEFAULT_ACTION_DIM);
            double reward = random.nextDouble() * 2.0 - 1.0; // 随机奖励在-1到1之间
            double[] nextState = generateRandomState(DEFAULT_STATE_DIM);
            boolean done = random.nextDouble() < 0.1; // 10%的概率结束
            
            // 选择算法
            String algorithm = ALGORITHMS[i % ALGORITHMS.length];
            System.out.println("训练请求 " + requestId + " 使用算法: " + algorithm);
            
            pythonInterface.trainAsync(state, action, reward, nextState, done, 
                    new EnhancedPythonInterface.Callback<Boolean>() {
                @Override
                public void onSuccess(Boolean result) {
                    System.out.println("训练请求 " + requestId + " 成功");
                    successCount.incrementAndGet();
                    activeRequests.decrementAndGet();
                    latch.countDown();
                }
                
                @Override
                public void onFailure(Exception e) {
                    System.out.println("训练请求 " + requestId + " 失败: " + e.getMessage());
                    activeRequests.decrementAndGet();
                    latch.countDown();
                }
            }, algorithm);
            
            // 小延迟以避免服务器过载
            if (i % 10 == 0) {
                Thread.sleep(50);
            }
        }
        
        // 等待所有请求完成
        if (!latch.await(numRequests * 500, TimeUnit.MILLISECONDS)) {
            System.out.println("警告: 部分请求未在预期时间内完成");
        }
        
        long endTime = System.currentTimeMillis();
        double elapsedSec = (endTime - startTime) / 1000.0;
        
        System.out.println("\n训练测试结果:");
        System.out.println("成功请求: " + successCount.get() + "/" + numRequests);
        System.out.println("总耗时: " + elapsedSec + " 秒");
        System.out.println("平均每秒处理: " + (numRequests / elapsedSec) + " 请求");
    }
    
    /**
     * 测试故障容错功能
     */
    private static void testFaultTolerance(EnhancedPythonInterface pythonInterface) 
            throws InterruptedException {
        System.out.println("\n===== 测试故障容错 =====");
        
        // 测试断开连接后的重新连接
        System.out.println("测试断开连接后的重新连接...");
        pythonInterface.disconnect();
        Thread.sleep(1000);
        
        boolean reconnected = pythonInterface.connect();
        System.out.println("重新连接结果: " + (reconnected ? "成功" : "失败"));
        
        if (!reconnected) {
            System.out.println("重新连接失败，尝试再次连接...");
            reconnected = pythonInterface.connect();
            System.out.println("第二次重新连接结果: " + (reconnected ? "成功" : "失败"));
            
            if (!reconnected) {
                System.out.println("无法重新连接，跳过剩余故障容错测试");
                return;
            }
        }
        
        // 测试发送格式错误的请求
        System.out.println("测试在连接恢复后的功能...");
        
        // 获取动作作为测试
        CountDownLatch latch = new CountDownLatch(1);
        AtomicInteger successCount = new AtomicInteger(0);
        
        double[] state = generateRandomState(DEFAULT_STATE_DIM);
        
        // 测试使用每种算法
        for (String algorithm : ALGORITHMS) {
            System.out.println("测试算法 " + algorithm + " 在恢复连接后是否正常工作");
            
            pythonInterface.getActionAsync(state, new EnhancedPythonInterface.Callback<double[]>() {
                @Override
                public void onSuccess(double[] action) {
                    System.out.println("恢复连接后请求成功，算法: " + algorithm + "，动作: " + formatArray(action));
                    successCount.incrementAndGet();
                    latch.countDown();
                }
                
                @Override
                public void onFailure(Exception e) {
                    System.out.println("恢复连接后请求失败: " + e.getMessage());
                    latch.countDown();
                }
            }, algorithm);
            
            // 等待请求完成
            if (!latch.await(5000, TimeUnit.MILLISECONDS)) {
                System.out.println("警告: 请求在恢复连接后未在5秒内完成");
            }
        }
        
        System.out.println("故障容错测试完成，成功恢复: " + successCount.get() + "/" + ALGORITHMS.length);
    }
    
    /**
     * 测试性能 - 批量请求
     */
    private static void testPerformance(EnhancedPythonInterface pythonInterface, int numRequests, int concurrentRequests) 
            throws InterruptedException {
        System.out.println("\n===== 测试性能 - 批量请求 =====");
        
        // 创建倒计时锁
        CountDownLatch latch = new CountDownLatch(numRequests * 2); // 动作和训练请求总数
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger activeRequests = new AtomicInteger(0);
        
        long startTime = System.currentTimeMillis();
        
        // 发送交错的获取动作和训练请求
        for (int i = 0; i < numRequests; i++) {
            final int requestId = i;
            
            // 控制并发请求数
            while (activeRequests.get() >= concurrentRequests) {
                Thread.sleep(5);
            }
            
            activeRequests.incrementAndGet();
            
            // 创建随机数据
            double[] state = generateRandomState(DEFAULT_STATE_DIM);
            double[] action = generateRandomAction(DEFAULT_ACTION_DIM);
            double reward = random.nextDouble() * 2.0 - 1.0;
            double[] nextState = generateRandomState(DEFAULT_STATE_DIM);
            boolean done = random.nextDouble() < 0.1;
            
            // 轮流使用不同算法
            String algorithm = ALGORITHMS[i % ALGORITHMS.length];
            
            // 发送动作请求
            pythonInterface.getActionAsync(state, new EnhancedPythonInterface.Callback<double[]>() {
                @Override
                public void onSuccess(double[] result) {
                    successCount.incrementAndGet();
                    activeRequests.decrementAndGet();
                    latch.countDown();
                }
                
                @Override
                public void onFailure(Exception e) {
                    System.out.println("性能测试动作请求 " + requestId + " 失败: " + e.getMessage());
                    activeRequests.decrementAndGet();
                    latch.countDown();
                }
            }, algorithm);
            
            // 发送训练请求
            pythonInterface.trainAsync(state, action, reward, nextState, done, 
                    new EnhancedPythonInterface.Callback<Boolean>() {
                @Override
                public void onSuccess(Boolean result) {
                    successCount.incrementAndGet();
                    activeRequests.decrementAndGet();
                    latch.countDown();
                }
                
                @Override
                public void onFailure(Exception e) {
                    System.out.println("性能测试训练请求 " + requestId + " 失败: " + e.getMessage());
                    activeRequests.decrementAndGet();
                    latch.countDown();
                }
            }, algorithm);
        }
        
        // 等待所有请求完成，最多等待30秒
        boolean completed = latch.await(30, TimeUnit.SECONDS);
        
        long endTime = System.currentTimeMillis();
        double elapsedSec = (endTime - startTime) / 1000.0;
        
        System.out.println("\n性能测试结果:");
        System.out.println("所有请求是否完成: " + (completed ? "是" : "否"));
        System.out.println("成功请求: " + successCount.get() + "/" + (numRequests * 2));
        System.out.println("总耗时: " + elapsedSec + " 秒");
        System.out.println("平均每秒处理: " + ((numRequests * 2) / elapsedSec) + " 请求");
    }
    
    /**
     * 快速请求压力测试
     */
    private static void testStressWithRapidRequests(EnhancedPythonInterface pythonInterface, int numRequests) 
            throws InterruptedException {
        System.out.println("\n===== 快速请求压力测试 =====");
        
        // 创建倒计时锁
        CountDownLatch latch = new CountDownLatch(numRequests);
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger failureCount = new AtomicInteger(0);
        
        long startTime = System.currentTimeMillis();
        
        System.out.println("发送 " + numRequests + " 个快速连续请求...");
        
        // 创建预生成的状态数组以加快请求发送速度
        double[][] states = new double[numRequests][];
        for (int i = 0; i < numRequests; i++) {
            states[i] = generateRandomState(DEFAULT_STATE_DIM);
        }
        
        // 快速发送请求，不等待响应
        for (int i = 0; i < numRequests; i++) {
            final int requestId = i;
            String algorithm = ALGORITHMS[i % ALGORITHMS.length];
            
            pythonInterface.getActionAsync(states[i], new EnhancedPythonInterface.Callback<double[]>() {
                @Override
                public void onSuccess(double[] action) {
                    successCount.incrementAndGet();
                    latch.countDown();
                }
                
                @Override
                public void onFailure(Exception e) {
                    failureCount.incrementAndGet();
                    latch.countDown();
                }
            }, algorithm);
            
            // 非常短的延迟，几乎是连续发送
            if (i % 20 == 0) {
                Thread.sleep(1);
            }
        }
        
        System.out.println("所有请求已发送，等待响应...");
        
        // 等待所有请求完成，最多等待60秒
        boolean completed = latch.await(60, TimeUnit.SECONDS);
        
        long endTime = System.currentTimeMillis();
        double elapsedSec = (endTime - startTime) / 1000.0;
        
        System.out.println("\n压力测试结果:");
        System.out.println("所有请求是否完成: " + (completed ? "是" : "否"));
        System.out.println("成功请求: " + successCount.get());
        System.out.println("失败请求: " + failureCount.get());
        System.out.println("总耗时: " + elapsedSec + " 秒");
        System.out.println("平均每秒处理: " + (numRequests / elapsedSec) + " 请求");
        System.out.println("成功率: " + (successCount.get() * 100.0 / numRequests) + "%");
    }
    
    /**
     * 生成随机状态数组
     */
    private static double[] generateRandomState(int dimension) {
        double[] state = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            state[i] = random.nextDouble() * 2.0 - 1.0; // 在-1到1之间
        }
        return state;
    }
    
    /**
     * 生成随机动作数组
     */
    private static double[] generateRandomAction(int dimension) {
        double[] action = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            action[i] = random.nextDouble() * 2.0 - 1.0; // 在-1到1之间
        }
        return action;
    }
    
    /**
     * 格式化数组为字符串
     */
    private static String formatArray(double[] array) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < array.length; i++) {
            sb.append(String.format("%.3f", array[i]));
            if (i < array.length - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}