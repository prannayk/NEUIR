count=0
for i in `seq 1 $1`; do
	count=$(echo "$count+1" | bc)
	filename="/media/hdd/hdd/data_backup/results/$2/$3/$4/tweet_list_$count.txt"
#	../dataset/asonam/trec_eval -q -m all_trec ../dataset/asonam/Italy-need-avail-qrel $filename 
	l=$(../dataset/asonam/trec_eval -q -m all_trec ../dataset/asonam/Italy-need-avail-qrel $filename | grep "P_100" | head -n1 | cut -f3 )
	y=$(../dataset/asonam/trec_eval -q -m all_trec ../dataset/asonam/Italy-need-avail-qrel $filename | grep "map" | head -n1 | cut -f3 )
	m=$(../dataset/asonam/trec_eval -q -m all_trec ../dataset/asonam/Italy-need-avail-qrel $filename | grep "recall_1000" | head -n1 | cut -f3 )
	t=$(echo "2/((1/$l) + (1/$m))" | bc -l )
	echo "$l , $m : $t : $y"
done
