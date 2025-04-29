package edu.boun.edgecloudsim.utils;

/**
 * 扩展SimUtils类的方法，添加缺失的方法
 */
public class SimUtilsExtensions {
    
    /**
     * 生成输出文件夹
     * @return 输出文件夹路径
     */
    public static String generateOutputFolder(SimUtils simUtils) {
        // 实现生成输出文件夹的逻辑
        // 或返回一个默认路径
        return "sim_results/output_" + System.currentTimeMillis();
    }
}