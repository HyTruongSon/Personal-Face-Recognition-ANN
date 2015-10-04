package MyLib;

import java.lang.*;
import java.io.*;

public class BinaryFileReader {
	
	BufferedInputStream f;
	
	public boolean open(String name) throws IOException{
		try {
			f = new BufferedInputStream ( new FileInputStream(name) );
		}catch (IOException e){
			return false;
		}
		return true;
	}
	
	public int read() throws IOException{
		return f.read();
	}
	
	public int available() throws IOException{
		return f.available();
	}
	
	public void close() throws IOException{
		f.close();
	}
	
}