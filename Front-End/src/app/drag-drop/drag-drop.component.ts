import { Component } from '@angular/core';
import { NgxFileDropEntry, FileSystemFileEntry, FileSystemDirectoryEntry } from 'ngx-file-drop';
import { Router } from '@angular/router';
import { XmlComparatorService } from '../xml-comparator.service';
import {HttpClient} from "@angular/common/http";

@Component({
  selector: 'app-drag-drop',
  templateUrl: './drag-drop.component.html',
  styleUrls: ['./drag-drop.component.css']
})
export class DragDropComponent {

  constructor(private router: Router, private XmlService: XmlComparatorService, private http: HttpClient) {};

  public file: NgxFileDropEntry[] = [];
  showError: boolean = false;
  messageError: string = "";
  xmlContent: string[] = [];
  public imageUrl: string = '';
  public fileName: string = '';
  public flag: boolean = true;
  public image_array : number[][] = [];

  public dropped(files: NgxFileDropEntry[],index:number) {
    if (files[0].fileEntry.isFile && this.isFileAllowed(files[0].fileEntry.name)) {
      const fileEntry = files[0].fileEntry as FileSystemFileEntry;
      this.fileName = files[0].fileEntry.name;
      this.readImageFile(fileEntry);
      this.flag = false;
    } else {
      this.showError = true;
      this.messageError="O formato do ficheiro tem de ser jpg.";
    }
  }

  private readImageFile(fileEntry: FileSystemFileEntry): void {
    fileEntry.file((file: File) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(img, 0, 0);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const pixels = imageData.data;
            const rgbArray = [];
            for (let i = 0; i < pixels.length; i += 4) {
              const red = pixels[i];
              const green = pixels[i + 1];
              const blue = pixels[i + 2];
              rgbArray.push([red, green, blue]);
            }
            this.image_array = rgbArray;
          }
        };
        img.src = reader.result as string;
      };
      reader.readAsDataURL(file);
    });
  }



  isFileAllowed(fileName: string) {
    let isFileAllowed = false;
    const allowedFiles = ['.jpg','.JPG'];
    const regex = /(?:\.([^.]+))?$/;
    const extension = regex.exec(fileName);
    if (undefined !== extension && null !== extension) {
        for (const ext of allowedFiles) {
            if (ext === extension[0]) {
                isFileAllowed = true;
            }
        }
    }
    return isFileAllowed;
}

  public CloseModal(){
    this.showError=false;
  }

  public deleteFile(index:number){
    this.flag=true;
    this.imageUrl = "";
  }

  compareFilesJcu() {
    if (this.image_array.length == 0) return;

    const requestBody = {
      image: this.image_array,
    };
    this.http.post<any>('http://127.0.0.1:5000/image-crop-predict', requestBody)
        .subscribe(response => {
          console.log(response)
        }, error => {
          console.error('Error sending message:', error);
        });
  }
}
