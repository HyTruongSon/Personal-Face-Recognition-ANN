package MyLib;

import java.lang.*;
import java.io.*;

public class Enhance {

	static int mask[][] = { {1,2,1} , {2,4,2} , {1,2,1} };
	static int width, height;
	
	public static int get(int f[][],int x,int y){
		if ((x<0)||(x>=width)||(y<0)||(y>=height)) return 1;
		return (f[x][y]+1)%2;
	}
	
	public static int cal(int f[][],int x,int y){
		int i,j,u,v,res;
		res=0;
		for (i=-1;i<=1;i++)
			for (j=-1;j<=1;j++){
				u=x+i;
				v=y+j;
				res+=get(f,u,v)*mask[1+i][1+j];
			}
		return (res/16+1)%2;
	}
	
	public static void process(int f[][],int g[][],int w,int h){
		int x,y;
		width=w;
		height=h;
		for (x=0;x<width;x++)
			for (y=0;y<height;y++)
				g[x][y]=cal(f,x,y);
	}
	
	public static void process(int f[][],int w,int h){
		int x,y;
		width=w;
		height=h;
		int g[][] = new int [width][height];
		for (x=0;x<width;x++)
			for (y=0;y<height;y++)
				g[x][y]=cal(f,x,y);
		for (x=0;x<width;x++)
			for (y=0;y<height;y++) f[x][y]=g[x][y];
	}

}