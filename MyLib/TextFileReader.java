package MyLib;

import java.io.*;

public class TextFileReader {
	
	BufferedReader file;
	
	public void open(String name) throws IOException {
		file = new BufferedReader(new FileReader(name));
	}
	
	public String ReadLine() throws IOException {
		return file.readLine();
	}
	
	public void close() throws IOException {
		file.close();
	}
	
}
