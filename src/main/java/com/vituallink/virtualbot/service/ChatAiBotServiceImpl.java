package com.vituallink.virtualbot.service;


import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

@Service
public class ChatAiBotServiceImpl implements ChatAiBotService {

    private final ChatClient chatClient;


    public ChatAiBotServiceImpl(ChatClient.Builder chatClient) {
        this.chatClient = chatClient.build();
    }

    @Override
    public String chat(String question) {
        return chatClient.prompt()
                .user(question)
                .call()
                .content();
    }
}
