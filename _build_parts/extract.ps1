$f='D:\New folder0010\wakeword_ml\train_torch.py'
$l=Get-Content $f
$head = $l[0..171]   # imports .. end of augment(); stop before the broken stub
Set-Content -Path 'D:\New folder0010\train_head.py' -Value $head -Encoding UTF8
$score = $l[231..240]
Set-Content -Path 'D:\New folder0010\train_score.py' -Value $score -Encoding UTF8
"head lines: $($head.Count)  score lines: $($score.Count)"
"--- head tail ---"; $head[-3..-1]