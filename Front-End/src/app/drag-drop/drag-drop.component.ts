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
  public image : File|null = null;
  public formData: FormData = new FormData();
  public answer: String = '';
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
      this.image = file
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
    this.image = null;
  }

  compareFilesJcu(type: number) {
    if (this.image == null)
      return;


    this.formData = new FormData();
    // @ts-ignore
    this.formData.append('file', this.image);
    this.formData.append('type', type.toString());

    this.http.post<any>('http://127.0.0.1:5000/image-crop-predict', this.formData)
      .subscribe(response => {
        this.answer = response.crop.replace('___', ' - ').replace('_', ' ')
      }, error => {
        console.error('Error sending message:', error);
      });
  }
}
