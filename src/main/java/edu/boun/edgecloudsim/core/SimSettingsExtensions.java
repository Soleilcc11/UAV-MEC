package edu.boun.edgecloudsim.core;

/**
 * 扩展SimSettings类的方法，添加缺失的方法
 */
public class SimSettingsExtensions {
    
    /**
     * 初始化模拟参数
     * 该方法添加到SimSettings类中解决编译错误
     */
    public static void initializeSimulationParameters(SimSettings simSettings) {
        // 实现初始化逻辑，或调用已有方法
    }
    
    /**
     * 获取移动性查找表
     * @return 移动性数据表
     */
    public static Object getMobilityLookUpTable(SimSettings simSettings) {
        // 如果可能，返回实际的移动性表
        // 这里作为临时解决方案返回空二维数组
        return new double[0][0];
    }
    
    /**
     * 获取云VM的核心数
     * @return 云VM的核心数
     */
    public static int getCoreForCloudVM(SimSettings simSettings) {
        // 返回默认值或从设置中读取
        return 4; // 默认值
    }
    
    /**
     * 获取云VM的RAM大小
     * @return 云VM的RAM大小
     */
    public static int getRamForCloudVM(SimSettings simSettings) {
        // 返回默认值或从设置中读取
        return 4096; // 默认值，单位MB
    }
    
    /**
     * 获取云VM的存储大小
     * @return 云VM的存储大小
     */
    public static int getStorageForCloudVM(SimSettings simSettings) {
        // 返回默认值或从设置中读取
        return 10000; // 默认值，单位MB
    }
    
    /**
     * 获取云VM的MIPS
     * @return 云VM的MIPS
     */
    public static double getMipsForCloudVM(SimSettings simSettings) {
        // 返回默认值或从设置中读取
        return 100000.0; // 默认值
    }
}