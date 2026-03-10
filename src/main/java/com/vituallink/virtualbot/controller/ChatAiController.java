package com.vituallink.virtualbot.controller;

import com.vituallink.virtualbot.service.ChatAiBotService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/chat")
@RequiredArgsConstructor
@CrossOrigin
public class ChatAiController {

    private final ChatAiBotService chatAiBotService;

    @GetMapping("/ask")
    public String ask(String question) {
        return chatAiBotService.chat(question);
    }
}
