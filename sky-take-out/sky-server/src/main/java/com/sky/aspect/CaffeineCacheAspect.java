package com.sky.aspect;

import com.alibaba.druid.util.StringUtils;
import com.github.xiaoymin.knife4j.core.util.CollectionUtils;
import com.google.common.collect.Lists;
import com.sky.annotation.EnableCaffeineCache;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.Cache;
import org.springframework.cache.CacheManager;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Component;

import java.util.*;

@Aspect
@Component
@Slf4j
@Lazy
public class CaffeineCacheAspect {

    // 注入CacheManager
    @Autowired
    private CacheManager caffineCacheManager;
    @Pointcut("@annotation(com.sky.annotation.EnableCaffeineCache)")
    public void caffeineCachePointCut(){}


    /**
     ** 使用around方法对入参及返回进行拦截
     */
    @Around("caffeineCachePointCut() && @annotation(caffeinCacheAnnotation)")
    public Object injectCaffeinCache(ProceedingJoinPoint proceedingJoinPoint, EnableCaffeineCache caffeinCacheAnnotation) throws Throwable {
        String cacheName = caffeinCacheAnnotation.value();
        Object[] args = proceedingJoinPoint.getArgs();
        Object param = null;
        if(args.length==1){
            param = args[0];
        }else{
            // 入参不合法，不开启缓存，直接执行原来的方法
            return proceedingJoinPoint.proceed(args);
        }
        try{
            Cache cache = caffineCacheManager.getCache(cacheName);

            // 如果是集合，批量查询redis
            if(param instanceof Collection){
                // 注意 key value 的有序性
                Collection collection = (Collection) param;
                Map<String,String> resultMap = new LinkedHashMap<>();
                List<String> rediskeys = Lists.newArrayList();
                // 遍历集合，先去内存缓存查询,没有的 存到redis Map
                for(Object key:collection){
                    String value = cache.get(key, String.class);
                    resultMap.put((String) key,value);
                    if (value == null) {
                        rediskeys.add((String)key);
                    }
                }
                // 去redis里面查询，执行原来的方法
                List<String> redisValues = (List<String>) proceedingJoinPoint.proceed(new Collection[]{rediskeys});

                // 将redis结果缓存到内存缓存中
                if(!CollectionUtils.isEmpty(redisValues)){
                    for(int i=0;i<rediskeys.size();i++){
                        String value = redisValues.get(i);
                        resultMap.put(rediskeys.get(i),value);
                        if(!StringUtils.isEmpty(value)){
                            cache.put(rediskeys.get(i),value);
                        }
                    }
                }
                // 最后返回 result
                Collection<String> values = resultMap.values();
                return new ArrayList<>(values);
            }else if (param instanceof String){
                // 如果是字符串
                String s = cache.get(param, String.class);
                if(s!=null){
                    return s;
                }else{
                    String cacheValue= (String) proceedingJoinPoint.proceed(args);
                    if(cacheValue!=null){
                        cache.put(param,cacheValue);
                    }
                    return cacheValue;
                }
            }
        }catch (Exception e){
            log.error("get cache error:{}",e);
        }finally {
            return proceedingJoinPoint.proceed(args);
        }
    }
}
