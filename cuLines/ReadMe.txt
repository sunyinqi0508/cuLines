Preporc:
	
	segments --hashtable--> table_contents, layout: table 0:[[fingerprint2s with same fingerprint1 in table 0(one fp2 only appears once)] [fp2s with fp1[1]]] table 1:[[] [] []] ...
	segment numbers put in segment_indices matching layout of table_contents recording segment range in segment_in_table
	segment_in_table expends corresponding segments in table_contents 

Lv1 search:
	Points --hashing--> segments that share same fingerprints finding their positions on segments_in_table
	
Lv2 search:
	Points 