import { Component, AfterViewInit } from '@angular/core';

interface Message {
  sender: string;
  content: string;
}

@Component({
  selector: 'app-chatbot',
  templateUrl: './chatbot.component.html',
  styleUrls: ['./chatbot.component.css']
})
export class ChatbotComponent implements AfterViewInit {

  messages: Message[] = [];
  userInput: string = '';

  constructor() { }

  ngAfterViewInit() {
    this.addBotMessage('Hi there 👋<br>How can I help you today?');
  }

  sendMessage() {
    if (this.userInput.trim() === '') return;

    this.addUserMessage(this.userInput);
    // You can replace this with actual chatbot logic to generate responses
    this.addBotMessage('Sorry, I am a simple chatbot and can\'t respond intelligently.');
    this.userInput = '';

    // Scroll to the bottom of the chatbox
    setTimeout(() => {
      const chatbox = document.querySelector('.chatbox');
      if (chatbox) {
        chatbox.scrollTop = chatbox.scrollHeight;
      }
    }, 0);
  }

  addUserMessage(message: string) {
    this.messages.push({ sender: 'You', content: message });
  }
  
  addBotMessage(message: string) {
    this.messages.push({ sender: 'Bot', content: message });
  }

  attachFile(){
    
  }
}
