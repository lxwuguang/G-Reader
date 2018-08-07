package competitions.duReader.bean;

/**
 * Copyright 2018 bejson.com 
 */
import java.util.List;

/**
* Auto-generated: 2018-03-26 11:43:50
*
* @author bejson.com (i@bejson.com)
* @website http://www.bejson.com/java2pojo/
*/
public class Pojo {

   private List<Document> documents;
   private List<List<Integer>> answer_spans;
   private List<String> fake_answers;
   private String question;
   private List<List<String>> segmented_answers;
   private List<String> answers;
   private List<Integer> answer_docs;
   private List<String> segmented_question;
   private String question_type;
   private long question_id;
   private String fact_or_opinion;
   private List<Double> match_scores;
   public void setDocuments(List<Document> documents) {
        this.documents = documents;
    }
    public List<Document> getDocuments() {
        return documents;
    }

   public void setAnswer_spans(List<List<Integer>> answer_spans) {
        this.answer_spans = answer_spans;
    }
    public List<List<Integer>> getAnswer_spans() {
        return answer_spans;
    }

   public void setFake_answers(List<String> fake_answers) {
        this.fake_answers = fake_answers;
    }
    public List<String> getFake_answers() {
        return fake_answers;
    }

   public void setQuestion(String question) {
        this.question = question;
    }
    public String getQuestion() {
        return question;
    }

   public void setSegmented_answers(List<List<String>> segmented_answers) {
        this.segmented_answers = segmented_answers;
    }
    public List<List<String>> getSegmented_answers() {
        return segmented_answers;
    }

   public void setAnswers(List<String> answers) {
        this.answers = answers;
    }
    public List<String> getAnswers() {
        return answers;
    }

   public void setAnswer_docs(List<Integer> answer_docs) {
        this.answer_docs = answer_docs;
    }
    public List<Integer> getAnswer_docs() {
        return answer_docs;
    }

   public void setSegmented_question(List<String> segmented_question) {
        this.segmented_question = segmented_question;
    }
    public List<String> getSegmented_question() {
        return segmented_question;
    }

   public void setQuestion_type(String question_type) {
        this.question_type = question_type;
    }
    public String getQuestion_type() {
        return question_type;
    }

   public void setQuestion_id(long question_id) {
        this.question_id = question_id;
    }
    public long getQuestion_id() {
        return question_id;
    }

   public void setFact_or_opinion(String fact_or_opinion) {
        this.fact_or_opinion = fact_or_opinion;
    }
    public String getFact_or_opinion() {
        return fact_or_opinion;
    }

   public void setMatch_scores(List<Double> match_scores) {
        this.match_scores = match_scores;
    }
    public List<Double> getMatch_scores() {
        return match_scores;
    }

}