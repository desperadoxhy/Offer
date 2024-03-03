package com.sky.properties;

public class CaffineCacheProperties {
    /**
     * cache队列最大大小
     */
    private int maxSize = 1000;

    /**
     * 过期时间，单位s
     */
    public int timeoutSecond = 6000;

    /**
     * cache名字
     */
    public String cacheName = "default";
}
