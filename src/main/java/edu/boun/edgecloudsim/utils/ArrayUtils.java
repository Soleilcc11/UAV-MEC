package edu.boun.edgecloudsim.utils;

/**
 * 数组工具类，提供安全的数组转换和操作方法
 */
public class ArrayUtils {
    
    /**
     * 安全地将Object转换为double数组
     * @param obj 要转换的对象
     * @return 转换后的double数组，如果对象不是double数组则返回null
     */
    public static double[] safeDoubleArray(Object obj) {
        if (obj == null) return null;
        if (obj instanceof double[]) return (double[])obj;
        return null;
    }
    
    /**
     * 安全地将Object转换为double二维数组
     * @param obj 要转换的对象
     * @return 转换后的double二维数组，如果对象不是double二维数组则返回null
     */
    public static double[][] safeDouble2DArray(Object obj) {
        if (obj == null) return null;
        if (obj instanceof double[][]) return (double[][])obj;
        return null;
    }
    
    /**
     * 安全地将Object转换为int数组
     * @param obj 要转换的对象
     * @return 转换后的int数组，如果对象不是int数组则返回null
     */
    public static int[] safeIntArray(Object obj) {
        if (obj == null) return null;
        if (obj instanceof int[]) return (int[])obj;
        return null;
    }
    
    /**
     * 安全地将Object转换为long数组
     * @param obj 要转换的对象
     * @return 转换后的long数组，如果对象不是long数组则返回null
     */
    public static long[] safeLongArray(Object obj) {
        if (obj == null) return null;
        if (obj instanceof long[]) return (long[])obj;
        return null;
    }
    
    /**
     * 安全地获取Object的数组长度
     * @param obj 要获取长度的对象
     * @return 数组的长度，如果对象不是数组则返回0
     */
    public static int safeArrayLength(Object obj) {
        if (obj == null) return 0;
        if (obj instanceof Object[]) return ((Object[])obj).length;
        if (obj instanceof double[]) return ((double[])obj).length;
        if (obj instanceof int[]) return ((int[])obj).length;
        if (obj instanceof long[]) return ((long[])obj).length;
        if (obj instanceof float[]) return ((float[])obj).length;
        if (obj instanceof boolean[]) return ((boolean[])obj).length;
        if (obj instanceof byte[]) return ((byte[])obj).length;
        if (obj instanceof char[]) return ((char[])obj).length;
        if (obj instanceof short[]) return ((short[])obj).length;
        return 0;
    }
    
    /**
     * 安全地获取二维数组的行数
     * @param obj 要获取行数的对象
     * @return 二维数组的行数，如果对象不是二维数组则返回0
     */
    public static int safe2DArrayRows(Object obj) {
        if (obj == null) return 0;
        if (obj instanceof Object[][]) return ((Object[][])obj).length;
        if (obj instanceof double[][]) return ((double[][])obj).length;
        if (obj instanceof int[][]) return ((int[][])obj).length;
        return 0;
    }
    
    /**
     * 安全地获取二维数组指定行的列数
     * @param obj 要获取列数的对象
     * @param row 行索引
     * @return 指定行的列数，如果对象不是二维数组或行索引无效则返回0
     */
    public static int safe2DArrayColumns(Object obj, int row) {
        if (obj == null) return 0;
        
        int rows = safe2DArrayRows(obj);
        if (rows <= 0 || row < 0 || row >= rows) return 0;
        
        if (obj instanceof Object[][]) {
            Object[] rowArray = ((Object[][])obj)[row];
            return rowArray != null ? rowArray.length : 0;
        }
        
        if (obj instanceof double[][]) {
            double[] rowArray = ((double[][])obj)[row];
            return rowArray != null ? rowArray.length : 0;
        }
        
        if (obj instanceof int[][]) {
            int[] rowArray = ((int[][])obj)[row];
            return rowArray != null ? rowArray.length : 0;
        }
        
        return 0;
    }
    
    /**
     * 安全地访问Object数组的元素
     * @param obj 数组对象
     * @param index 索引
     * @return 数组元素，如果对象不是数组或索引无效则返回null
     */
    public static Object safeArrayGet(Object obj, int index) {
        if (obj == null) return null;
        if (!(obj.getClass().isArray())) return null;
        
        int length = safeArrayLength(obj);
        if (index < 0 || index >= length) return null;
        
        if (obj instanceof Object[]) return ((Object[])obj)[index];
        
        // 对于基本类型数组，返回包装类型
        if (obj instanceof double[]) return ((double[])obj)[index];
        if (obj instanceof int[]) return ((int[])obj)[index];
        if (obj instanceof long[]) return ((long[])obj)[index];
        if (obj instanceof float[]) return ((float[])obj)[index];
        if (obj instanceof boolean[]) return ((boolean[])obj)[index];
        if (obj instanceof byte[]) return ((byte[])obj)[index];
        if (obj instanceof char[]) return ((char[])obj)[index];
        if (obj instanceof short[]) return ((short[])obj)[index];
        
        return null;
    }
    
    /**
     * 安全地访问二维数组的元素
     * @param obj 二维数组对象
     * @param row 行索引
     * @param col 列索引
     * @return 数组元素，如果对象不是二维数组或索引无效则返回null
     */
    public static Object safe2DArrayGet(Object obj, int row, int col) {
        if (obj == null) return null;
        
        int rows = safe2DArrayRows(obj);
        if (rows <= 0 || row < 0 || row >= rows) return null;
        
        int cols = safe2DArrayColumns(obj, row);
        if (cols <= 0 || col < 0 || col >= cols) return null;
        
        if (obj instanceof Object[][]) return ((Object[][])obj)[row][col];
        if (obj instanceof double[][]) return ((double[][])obj)[row][col];
        if (obj instanceof int[][]) return ((int[][])obj)[row][col];
        
        return null;
    }
    
    /**
     * 将对象转换为double值
     * @param obj 要转换的对象
     * @param defaultValue 默认值，当转换失败时返回
     * @return 转换后的double值
     */
    public static double toDouble(Object obj, double defaultValue) {
        if (obj == null) return defaultValue;
        
        try {
            if (obj instanceof Number) return ((Number)obj).doubleValue();
            if (obj instanceof String) return Double.parseDouble((String)obj);
        } catch (Exception e) {
            return defaultValue;
        }
        
        return defaultValue;
    }
    
    /**
     * 将对象转换为int值
     * @param obj 要转换的对象
     * @param defaultValue 默认值，当转换失败时返回
     * @return 转换后的int值
     */
    public static int toInt(Object obj, int defaultValue) {
        if (obj == null) return defaultValue;
        
        try {
            if (obj instanceof Number) return ((Number)obj).intValue();
            if (obj instanceof String) return Integer.parseInt((String)obj);
        } catch (Exception e) {
            return defaultValue;
        }
        
        return defaultValue;
    }
    
    /**
     * 将对象转换为long值
     * @param obj 要转换的对象
     * @param defaultValue 默认值，当转换失败时返回
     * @return 转换后的long值
     */
    public static long toLong(Object obj, long defaultValue) {
        if (obj == null) return defaultValue;
        
        try {
            if (obj instanceof Number) return ((Number)obj).longValue();
            if (obj instanceof String) return Long.parseLong((String)obj);
        } catch (Exception e) {
            return defaultValue;
        }
        
        return defaultValue;
    }
}