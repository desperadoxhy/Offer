package com.sky.config;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.sky.properties.CaffineCacheProperties;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.cache.CacheManager;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.cache.caffeine.CaffeineCache;
import org.springframework.cache.caffeine.CaffeineCacheManager;
import org.springframework.cache.support.SimpleCacheManager;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

@Configuration
@EnableCaching
public class CaffineCacheConfig {

    /**
     * cache名称
     * 提示：@Primary，用于标识一个Bean（组件）是首选的候选者。当有多个同类型的Bean（组件）时，使用了@Primary注解的Bean将会成为默认选择，
     * 如果没有其他限定符（如@Qualifier）指定具体要使用的Bean，则会优先选择带有@Primary注解的Bean。
     * @return
     */
    @Primary
    @Bean("defaultCacheManager")
    public CacheManager cacheManagerOne(){
        CaffeineCacheManager cacheManager = new CaffeineCacheManager();
        //Caffeine配置
        Caffeine<Object, Object> caffeine = Caffeine.newBuilder()
                //最后一次写入后经过固定时间过期
                .expireAfterWrite(10, TimeUnit.SECONDS)
                //maximumSize=[long]: 缓存的最大条数
                .maximumSize(1000);
        cacheManager.setCaffeine(caffeine);
        return cacheManager;
    }

    /**
     * cache名称
     * 缓存一些零散数据，根据缓存时间不同可能需要配置不同的bean方法
     * @return
     */
    @Bean("cacheManagertwo")
    public CacheManager cacheManageTwo(){
        CaffeineCacheManager cacheManager = new CaffeineCacheManager();
        //Caffeine配置
        Caffeine<Object, Object> caffeine = Caffeine.newBuilder()
                //最后一次写入后经过固定时间过期
                .expireAfterWrite(30, TimeUnit.SECONDS)
                //maximumSize=[long]: 缓存的最大条数
                .maximumSize(50);
        cacheManager.setCaffeine(caffeine);
        return cacheManager;
    }


}
