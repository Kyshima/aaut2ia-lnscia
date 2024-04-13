import { Component, AfterViewInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

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

  private apiUrl = 'http://localhost:5000/chat'; // Replace with your API URL
  private giveDescription = "no"
  private give_treatment = "no"

  constructor(private http: HttpClient) { }

  ngAfterViewInit() {
    this.addBotMessage('Hi there 👋<br>How can I help you today?');
  }

  sendMessage() {
    if (this.userInput.trim() === '') return;

    this.addUserMessage(this.userInput);
    // You can replace this with actual chatbot logic to generate responses
    //this.addBotMessage('Sorry, I am a simple chatbot and can\'t respond intelligently.');

    const requestBody = {
      text: this.userInput,
      give_treatment: this.give_treatment // Assuming you don't always want to give treatment
    };

    this.http.post<any>('http://localhost:5001/chat', requestBody)
    .subscribe(response => {
      const botResponse = response.response;
      this.giveDescription = response.give_description;

      this.addBotMessage(botResponse);

      if (this.giveDescription === 'yes') {
        this.give_treatment = "yes";
      }else {
        this.give_treatment = "no";
      }
    }, error => {
      console.error('Error sending message:', error);
    });

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
