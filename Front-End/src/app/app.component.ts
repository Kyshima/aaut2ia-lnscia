import { Component } from '@angular/core';
import { Router } from '@angular/router';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  confirmation : boolean = false;

  constructor(public router: Router){};

  HomePage() {
    this.CloseModal()
    this.router.navigate(['/']);
  }

  ShowModal() {
    if(this.router.url != "/"){
      this.confirmation = true;
    }else{
      this.HomePage();
    }
  }

  public CloseModal(){
    this.confirmation = false;
  }

  public NavigateChatBot(){
    this.router.navigate(['/chatbot']);
  }

}
