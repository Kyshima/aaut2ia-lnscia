import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { FileXML } from './FileXml';
import { environment } from 'src/environments/environment';
import * as Diff2Html from 'diff2html';
import * as jsDiff from 'diff';
import { OutputFormatType } from 'diff2html/lib/types';
import { saveAs } from 'file-saver';

@Injectable({
  providedIn: 'root'
})
export class XmlComparatorService {
  private storageKey = 'xmlContent';
  xmlContent: string;
  
  constructor(private http: HttpClient) {
    const storedData = localStorage.getItem(this.storageKey);
    this.xmlContent = storedData ? JSON.parse(storedData) : [];
  }

  SaveFiles(files: string) {
    this.xmlContent = files;
    localStorage.setItem(this.storageKey, JSON.stringify(this.xmlContent));
  }

  ReadFiles() {
    return this.xmlContent;
  }

  ReadXML1() {
    const storedValue = localStorage.getItem(this.storageKey);

    return storedValue ? JSON.parse(storedValue) : null;
  }

}
