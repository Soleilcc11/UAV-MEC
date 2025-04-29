package edu.boun.edgecloudsim.utils;

/**
 * 实用工具类，用于解决EdgeCloudSim中的数组转换问题
 */
public class ArrayUtils {

    /**
     * 将Object安全转换为double[][]
     * 
     * @param obj 需要转换的对象
     * @return 转换后的double[][]，如果无法转换则返回null
     */
    public static double[][] toDouble2DArray(Object obj) {
        if (obj instanceof double[][]) {
            return (double[][]) obj;
        }
        return new double[0][0]; // 返回空数组而不是null，避免空指针
    }
    
    /**
     * 将Object安全转换为double[]
     * 
     * @param obj 需要转换的对象
     * @return 转换后的double[]，如果无法转换则返回null
     */
    public static double[] toDoubleArray(Object obj) {
        if (obj instanceof double[]) {
            return (double[]) obj;
        }
        return new double[0]; // 返回空数组而不是null
    }
    
    /**
     * 将Object安全转换为int[]
     * 
     * @param obj 需要转换的对象
     * @return 转换后的int[]，如果无法转换则返回null
     */
    public static int[] toIntArray(Object obj) {
        if (obj instanceof int[]) {
            return (int[]) obj;
        }
        return new int[0]; // 返回空数组而不是null
    }
    
    /**
     * 将Object安全转换为String[]
     * 
     * @param obj 需要转换的对象
     * @return 转换后的String[]，如果无法转换则返回null
     */
    public static String[] toStringArray(Object obj) {
        if (obj instanceof String[]) {
            return (String[]) obj;
        }
        return new String[0]; // 返回空数组而不是null
    }
    
    /**
     * 获取Object作为数组的长度
     * 
     * @param obj 需要获取长度的对象
     * @return 对象作为数组的长度，如果不是数组则返回0
     */
    public static int length(Object obj) {
        if (obj == null) {
            return 0;
        }
        if (obj.getClass().isArray()) {
            return java.lang.reflect.Array.getLength(obj);
        }
        return 0;
    }
    
    /**
     * 安全地从2D数组获取值
     * 
     * @param array 2D数组对象
     * @param row 行索引
     * @param col 列索引
     * @param defaultValue 默认值
     * @return 指定位置的值，如果访问失败则返回默认值
     */
    public static double getValue(Object array, int row, int col, double defaultValue) {
        if (array == null) {
            return defaultValue;
        }
        
        try {
            if (array instanceof double[][]) {
                return ((double[][]) array)[row][col];
            }
        } catch (Exception e) {
            // 发生任何异常，返回默认值
        }
        
        return defaultValue;
    }
    
    /**
     * 安全地从对象获取并转换为double[][]后访问特定元素
     * 这个方法简化了常见的"从SimSettings获取LookUpTable并访问特定元素"的模式
     * 
     * @param obj 源对象
     * @param row 行索引
     * @param col 列索引
     * @return 转换后访问的double值
     */
    public static double getDoubleValueFromTable(Object obj, int row, int col) {
        try {
            return ((double[][]) obj)[row][col];
        } catch (Exception e) {
            return 0.0;
        }
    }
    
    /**
     * 获取并转换为long的值，特别用于从double数组获取长整型值
     * 
     * @param obj 源对象
     * @param row 行索引
     * @param col 列索引  
     * @return 转换为long的值
     */
    public static long getLongValueFromTable(Object obj, int row, int col) {
        try {
            return (long) ((double[][]) obj)[row][col];
        } catch (Exception e) {
            return 0L;
        }
    }
    
    /**
     * 获取并转换为int的值，特别用于从double数组获取整型值
     * 
     * @param obj 源对象
     * @param row 行索引
     * @param col 列索引
     * @return 转换为int的值
     */
    public static int getIntValueFromTable(Object obj, int row, int col) {
        try {
            return (int) ((double[][]) obj)[row][col];
        } catch (Exception e) {
            return 0;
        }
    }
    
    /**
     * 安全解析字符串为整数，如果解析失败则返回默认值
     * 
     * @param str 要解析的字符串
     * @param defaultValue 默认值
     * @return 解析后的整数，如果解析失败则返回默认值
     */
    public static int parseIntSafely(String str, int defaultValue) {
        try {
            return Integer.parseInt(str);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
}