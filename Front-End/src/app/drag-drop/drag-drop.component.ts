import { Component } from '@angular/core';
import { NgxFileDropEntry, FileSystemFileEntry, FileSystemDirectoryEntry } from 'ngx-file-drop';
import { Router } from '@angular/router';
import { XmlComparatorService } from '../xml-comparator.service';


@Component({
  selector: 'app-drag-drop',
  templateUrl: './drag-drop.component.html',
  styleUrls: ['./drag-drop.component.css']
})
export class DragDropComponent {

  constructor(private router: Router, private XmlService: XmlComparatorService) {};

  public file: NgxFileDropEntry[] = [];
  showError: boolean = false;
  messageError: string = "";
  xmlContent: string[] = [];
  public imageUrl: string = '';
  public fileName: string = '';
  public flag: boolean = true;

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
        this.imageUrl = reader.result as string || '';
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
    if (this.flag){  
      this.XmlService.SaveFiles(this.imageUrl);
    }else {
      this.showError = true;
      this.messageError="É necessário importar 2 ficheiros";
    }
  } 
}
