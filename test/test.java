package test;
import java.lang.Runtime;
import java.lang.Exception;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
 
public class test {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		String filePath = "C:\\Users\\User\\Desktop\\model\\test\\test_ex.bat";
		try {
		     
		    Process p = Runtime.getRuntime().exec(filePath);
		    p.waitFor();
		    
		    InputStream in = p.getInputStream();
		    ByteArrayOutputStream baos = new ByteArrayOutputStream();
		     
		    int c = -1;
		    while((c = in.read()) != -1)
		    {
		        baos.write(c);
		    }
		     
		    String response = new String(baos.toByteArray());
		    System.out.println("Response From Exe : "+response);
		    
		     
		} catch (Exception e) {
		    e.printStackTrace();
		}

	}
}
