package competitions.duReader.bean;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TfIdf implements Serializable {
	/**
	* 
	*/
	private static final long serialVersionUID = -5052815188356320024L;
	// 存放（单词，单词数量）
	public HashMap<String, Integer> dict = new HashMap<String, Integer>();
	// 存放（单词，单词词频）
	public HashMap<String, Float> tf = new HashMap<String, Float>();
	public HashMap<String, Integer> dt = new HashMap<String, Integer>();

	public HashMap<String, Float> idf = new HashMap<String, Float>();
	public int wordCount = 0;
	public int D = 0;

	/**
	 * 计算每个文档的tf值
	 * 
	 * @param wordAll
	 * @return Map<String,Float> key是单词 value是tf值
	 */
	public Map<String, Float> tfCalculate() {
		for (Map.Entry<String, Integer> entry : dict.entrySet()) {
			float wordTf = (float) entry.getValue() / wordCount;
			tf.put(entry.getKey(), wordTf);
		}
		return tf;
	}

	public void countWord(List<String> para) {
		/**
		 * 统计每个单词的数量，并存放到map中去 便于以后计算每个单词的词频 单词的tf=该单词出现的数量n/总的单词数wordCount
		 */
		for (String word : para) {
			wordCount++;
			if (dict.containsKey(word)) {
				dict.put(word, dict.get(word) + 1);
			} else {
				dict.put(word, 1);
			}
		}
	}

	/**
	 * 
	 * @param D
	 *            总文档数
	 * @param doc_words
	 *            每个文档对应的分词
	 * @param tf
	 *            计算好的tf,用这个作为基础计算tfidf
	 * @return 每个文档中的单词的tfidf的值
	 * @throws IOException
	 * @throws FileNotFoundException
	 */
	public void tfidfCount(List<String> words) {
		D++;
		HashMap<String, Integer> a = new HashMap<>();
		for (String i : words)
			a.put(i, 1);
		for (String key : a.keySet()) {
			if (dt.containsKey(key)) {
				dt.put(key, dt.get(key) + 1);
			} else {
				dt.put(key, 1);
			}
		}
	}

	public Map<String, Float> idfCalculate() {
		for (String key : tf.keySet()) {
			float idfvalue = (float) Math.log(Float.valueOf(D) / dt.get(key));
			idf.put(key, idfvalue);
		}
		return idf;
	}

	public void write() throws FileNotFoundException, IOException {
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("tfidfzhidao.txt"));
		oos.writeObject(this);
		oos.close();
	}

	public static TfIdf read() throws FileNotFoundException, IOException, ClassNotFoundException {
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream("tfidfzhidao.txt"));
		TfIdf tfidf = (TfIdf) ois.readObject();
		ois.close();
		return tfidf;

	}

}
