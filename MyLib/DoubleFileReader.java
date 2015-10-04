package MyLib;

import java.lang.*;
import java.io.*;

public class DoubleFileReader {
	
	BufferedInputStream f;
	
	public void open(String name) throws IOException{
		f = new BufferedInputStream ( new FileInputStream(name) );
	}
	
	public double read() throws IOException{
		int v;
		char ch;
		String s;
		s="";
		while (true){
			v=f.read();
			ch=(char)(v);
			if (ch!=' '){
				s+=ch;
				break;
			}
		}
		while (true){
			v=f.read();
			if (v==-1) break;
			ch=(char)(v);
			if (ch==' ') break;
			s+=ch;
		}
		return Double.parseDouble(s);
	}
	
	public void close() throws IOException{
		f.close();
	}
	
}