from datetime import datetime,timezone
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import pandas as pd
from src.current import kaggle_queue
from src.current.kaggle_queue import count_today,duplicate,validate


class SubmissionQueueGuards(unittest.TestCase):
    def test_daily_quota_uses_utc_date(self):
        rows=[{'date':'2026-11-07T23:59:59Z'},{'date':'2026-11-08T00:00:01Z'}]
        self.assertEqual(count_today(rows,datetime(2026,11,8,0,0,2,tzinfo=timezone.utc)),1)

    def test_no_submission_before_reset_even_with_empty_registry(self):
        with self.assertRaisesRegex(ValueError,'reset has not occurred'):
            validate({'not_before_utc':'2026-11-08T00:00:00Z'},datetime(2026,11,7,23,59,tzinfo=timezone.utc),{'submissions':[]},[])

    def test_duplicate_by_digest_or_remote_filename(self):
        row={'description':'approved sha256 abc123','fileName':'candidate.csv'}
        self.assertIs(duplicate([row],'abc123','other.csv'),row)
        self.assertIs(duplicate([row],'different','candidate.csv'),row)
        self.assertIsNone(duplicate([row],'different','other.csv'))

    def test_complete_offline_fixture_uploads_once_and_duplicate_run_fails(self):
        class FakeAPI:
            uploads=0
            def authenticate(self):pass
            def competition_submissions(self,competition):
                return [] if not self.uploads else [{'ref':123,'publicScore':'777000','status':'COMPLETE'}]
            def competition_submit(self,filename,message,competition):
                self.uploads+=1;return '{"ref":123}'
        fake=FakeAPI();module=types.ModuleType('kaggle.api.kaggle_api_extended');module.KaggleApi=lambda:fake
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for folder in ['reports/current','data/raw/kaggle','logs']:(root/folder).mkdir(parents=True,exist_ok=True)
            pd.DataFrame({'index':range(597),'SI VACV':[1.]*597}).to_csv(root/'candidate.csv',index=False)
            pd.DataFrame({'index':range(597)}).to_csv(root/'data/raw/kaggle/sample_submit.csv',index=False)
            (root/'artifact.bin').write_bytes(b'fixture only')
            (root/'verify.json').write_text(json.dumps({'status':'passed','checks':[{'candidate':'fixture','passed':True}]}))
            (root/'reports/current/kaggle_goal_submissions.json').write_text(json.dumps({'submissions':[]}))
            plan={'candidate':'fixture','submission':'candidate.csv','artifact':'artifact.bin','verification':'verify.json',
                'not_before_utc':'2026-11-08T00:00:00Z','expires_utc':'2026-11-08T00:15:00Z','max_daily_submissions':20,
                'state_file':'state.json','cancel_file':'logs/cancel','mean_oof_mse':1.,'competition':'offline-fixture','message':'offline fixture'}
            for key in ['submission','artifact','verification']:plan[key+'_sha256']=hashlib.sha256((root/plan[key]).read_bytes()).hexdigest()
            path=root/'plan.json';path.write_text(json.dumps(plan))
            with patch.object(kaggle_queue,'ROOT',root),patch.object(kaggle_queue,'now',return_value=datetime(2026,11,8,0,0,5,tzinfo=timezone.utc)),\
                 patch.dict(sys.modules,{'kaggle.api.kaggle_api_extended':module}),contextlib.redirect_stdout(io.StringIO()):
                kaggle_queue.run(path)
                with self.assertRaisesRegex(ValueError,'Already in submission registry'):kaggle_queue.run(path)
            self.assertEqual(fake.uploads,1)
            history=json.loads((root/'reports/current/kaggle_goal_submissions.json').read_text())
            self.assertEqual(len(history['submissions']),1)
            self.assertEqual(history['submissions'][0]['ref'],123)


if __name__=='__main__':unittest.main()
