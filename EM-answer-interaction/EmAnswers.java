package competitions.duReader.preprocess;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.alibaba.fastjson.JSON;

import competitions.duReader.bean.TfIdf;
import competitions.duReader.bean.Answer;
import competitions.duReader.bean.Document;
import competitions.duReader.bean.Pojo;

public class EmAnswers {
	public static List<List<String>> paras;

	public static List<List<String>> real_paras;
	public static Map<Integer, String> id2Word;
	public static Map<String, Integer> word2id;
	public static float[] wordValues;
	public static float[][] wordValueInPara;
	public static float[] paraProps;
	public static TfIdf tfidf;

	public static void main(String... args) throws FileNotFoundException, IOException, ClassNotFoundException {
		tfidf = TfIdf.read();
		run();
	}

	public static void em() {
		// 初始化EM参数
		int num = 0;
		id2Word = new HashMap<>();
		word2id = new HashMap<>();
		for (List<String> i : paras) {
			for (String j : i) {
				if (word2id.get(j) == null) {
					word2id.put(j, num);
					id2Word.put(num, j);
					num++;
				}
			}
		}
		paraProps = new float[paras.size()];
		wordValues = new float[num];
		wordValueInPara = new float[paras.size()][num];
		for (int i = 0; i < wordValues.length; i++) {
			if(tfidf.idf.get(id2Word.get(i))==null)
				tfidf.idf.put(id2Word.get(i),(float) 14.5);
			wordValues[i] = tfidf.idf.get(id2Word.get(i));
		}
		
		float[] paraValues = new float[paras.size()];
		float totalProp = (float) 0.0;
		for (int i = 0; i < paraValues.length; i++) {
			paraValues[i] = getParaValue(paras.get(i));
			totalProp += paraValues[i];
		}
		
		for (int i = 0; i < paraProps.length; i++) {
			paraProps[i] = paraValues[i] / totalProp;
		}

		// 开始em迭代
		for (int i = 0; i < 20; i++) {
			emStep();
		}
		
		paraValues = new float[paras.size()];
		totalProp = (float) 0.0;
		for (int i = 0; i < paraValues.length; i++) {
			paraValues[i] = getParaValue(paras.get(i))/paras.get(i).size();
			totalProp += paraValues[i];
		}
		// 归一化，根据其价值来求答案为该段落的概率
		for (int i = 0; i < paraProps.length; i++) {
			paraProps[i] = paraValues[i] / totalProp;
		}
	}

	public static void emStep() {
		// 求某个词在每个段落里的价值
		for (int i = 0; i < paras.size(); i++) {
			float totalInfoValue = 0;
			for (String word : paras.get(i)) {
				totalInfoValue += tfidf.idf.get(word);
			}
			for (String word : paras.get(i)) {
				int wordIndex = word2id.get(word);
				wordValueInPara[i][wordIndex] = tfidf.idf.get(word) * paraProps[i] / totalInfoValue;
			}
		}
		// 求每个词在整个答案集合中的价值
		for (int i = 0; i < wordValues.length; i++) {
			float totalWordValue = 0;
			for (int j = 0; j < paras.size(); j++) {
				totalWordValue += wordValueInPara[j][i];
			}
			wordValues[i] = totalWordValue / paras.size();
		}

		float[] paraValues = new float[paras.size()];

		float totalProp = (float) 0.0;
		// 求每一段落的价值
		for (int i = 0; i < paraValues.length; i++) {
			paraValues[i] = getParaValue(paras.get(i));
			totalProp += paraValues[i];
		}
		// 归一化，根据其价值来求答案为该段落的概率
		for (int i = 0; i < paraProps.length; i++) {
			paraProps[i] = paraValues[i] / totalProp;
		}
	}

	public static float getParaValue(List<String> para) {
		float paraValue = 0;
		for (String i : para) {
			paraValue += wordValues[word2id.get(i)];
		}
		return paraValue;
	}
	public static void getEmPojoParas(Pojo pojo) {
		paras = new ArrayList<List<String>>();
		real_paras = new ArrayList<List<String>>();
		for (Document i : pojo.getDocuments()) {
			HashMap<String,Integer> filter=new HashMap<String,Integer>();
			for (List<String> para : i.getSegmented_paragraphs()) {
				String sentence="";
				for (String j : para) sentence+=j;
				if(filter.get(sentence)!=null) continue;
				filter.put(sentence, 1);
				ArrayList<String> p = new ArrayList<String>();
				ArrayList<String> real_p = new ArrayList<String>();
				for (String word : para) {
					real_p.add(word);
					// 过滤标点符号
					if (word.replaceAll("\\pP", "").equals(""))
						continue;
					p.add(word);
				}
				if(p.size()==0) continue;
				paras.add(p);
				real_paras.add(real_p);
			}
		}
	}

	public static void run() throws FileNotFoundException, IOException, ClassNotFoundException {
		// JSON文件路径
		File json = new File("C:/Users/rongf/Desktop/devset/zhidao.dev.json");
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(json));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		String line = "";
		int num = 0;
		try {
			while ((line = br.readLine()) != null) // 读取到的内容给line变量
			{
				if (num > 1000)
					break;
				Pojo pojo = JSON.parseObject(line, Pojo.class);
				getEmPojoParas(pojo);
				em();
				System.out.println("问题是:" + pojo.getQuestion());
				int bestIndex=0;
				double score=0.0;
				for (int i = 0; i < paras.size(); i++) {
					if(paraProps[i]>score) {
						bestIndex=i;
						score=paraProps[i];
					}
					System.out.print("置信概率 " + paraProps[i] + " ，其答案为：");
					for (String j : real_paras.get(i)) 
						System.out.print(j);
					System.out.println();
				}
				System.out.print("最好答案为：");
				String answer="";
				for (String j : real_paras.get(bestIndex)) {
					answer+=j;
					System.out.print(j);
				}
				System.out.println();

				System.out.println("参考:" );
				for(String aAns:pojo.getAnswers()) {
					System.out.println(aAns);
				}
				num++;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		br.close();
	}
}
