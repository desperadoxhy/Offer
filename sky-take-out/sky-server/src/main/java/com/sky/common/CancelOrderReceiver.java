package com.sky.common;

import com.sky.service.OrderService;
import org.springframework.amqp.rabbit.annotation.RabbitHandler;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * 取消订单消息的处理者
 * Created by macro on 2018/9/14.
 */
@Component
@RabbitListener(queues = "mall.order.cancel")
public class CancelOrderReceiver {
    @Autowired
    private OrderService orderService;
    @RabbitHandler //表示下面的方法 handle 将会处理接收到的 RabbitMQ 消息
    public void handle(Long orderId) throws Exception {
        orderService.userCancelById(orderId);
    }
}
