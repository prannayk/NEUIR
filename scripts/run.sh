count=0
for i in `seq 1 $1`; do
	count=$(echo "$count+1" | bc)
	filename="../results/$2/$3/$4/tweet_list_$count.txt"
	l=$(../dataset/asonam/trec_eval -q -m all_trec ../dataset/asonam/Nepal-need-avail-qrel $filename | grep "Rprec" | head -n1 | cut -f3 )
	m=$(../dataset/asonam/trec_eval -q -m all_trec ../dataset/asonam/Nepal-need-avail-qrel $filename | grep "recall_1000" | head -n1 | cut -f3 )
	t=$(echo "2/((1/$l) + (1/$m))" | bc -l )
	echo "$l , $m : $t"
done
