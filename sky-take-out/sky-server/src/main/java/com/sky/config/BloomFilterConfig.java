package com.sky.config;

import com.google.common.hash.BloomFilter;
import com.google.common.hash.Funnels;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.nio.charset.Charset;

@Configuration
public class BloomFilterConfig {

    @Bean
    public BloomFilter<String> bloomFilter() {
        /**
         * 布隆过滤器
         * 三个参数 (Funnel<? super T> funnel, long expectedInsertions, double fpp)
         * Funnel<? super T> funnel : 数据类型 (long 保存商品ID)
         * long expectedInsertions : 希望插入值的个数
         * double fpp : 错误率，不设置默认为0.03 ，过小的话误判太多，造成数据库接受到大量请求，过大数组长度过长效率降低
         */
        return BloomFilter.create(Funnels.stringFunnel(Charset.defaultCharset()), 1000000, 0.03);
    }
}