package competitions.duReader.bean;

import java.util.List;

public class Document {

	    private boolean is_selected;
	    private String title;
	    private int most_related_para;
	    private List<String> segmented_title;
	    private List<List<String>> segmented_paragraphs;
	    private List<String> paragraphs;
	    private int bs_rank_pos;
	    public void setIs_selected(boolean is_selected) {
	         this.is_selected = is_selected;
	     }
	     public boolean getIs_selected() {
	         return is_selected;
	     }

	    public void setTitle(String title) {
	         this.title = title;
	     }
	     public String getTitle() {
	         return title;
	     }

	    public void setMost_related_para(int most_related_para) {
	         this.most_related_para = most_related_para;
	     }
	     public int getMost_related_para() {
	         return most_related_para;
	     }

	    public void setSegmented_title(List<String> segmented_title) {
	         this.segmented_title = segmented_title;
	     }
	     public List<String> getSegmented_title() {
	         return segmented_title;
	     }

	    public void setSegmented_paragraphs(List<List<String>> segmented_paragraphs) {
	         this.segmented_paragraphs = segmented_paragraphs;
	     }
	     public List<List<String>> getSegmented_paragraphs() {
	         return segmented_paragraphs;
	     }

	    public void setParagraphs(List<String> paragraphs) {
	         this.paragraphs = paragraphs;
	     }
	     public List<String> getParagraphs() {
	         return paragraphs;
	     }

	    public void setBs_rank_pos(int bs_rank_pos) {
	         this.bs_rank_pos = bs_rank_pos;
	     }
	     public int getBs_rank_pos() {
	         return bs_rank_pos;
	     }

	}
