����ʶ�𲿷ִ����˵���ĵ���
**************************************************�ο�����**********************************************************
*ʶ��ģ�Ͳ���CRNN���������ӣ�
*							https://arxiv.org/abs/1507.05717
*�ο���GitHub������ģ�͵�TensorFlowʵ�֣�����Ϊ��
*											https://github.com/MaybeShewill-CV/CRNN_Tensorflow   
*											https://github.com/bai-shang/crnn_ctc_ocr.Tensorflow
********************************************************************************************************************

�ļ��� char_map ����һ��json�ļ�char_map.json�����ڽ��ַ�ӳ��Ϊ��Ӧ���ֻ�����෴����
�ļ��� ckpt_log_save ����ѵ������ͼ�¼ 
�ļ��� config �а����ű�model_config.py��������ģ�͵Ĳ��ֳ�����
�ļ��� crnn_model �еĽű�crnn_net.py�б��湹��CRNN����Ĵ���
�ļ��� data_provider �а��������ű���write_tfrecord.py���ڽ�����תΪtfrecord�ļ���read_tfrecord.py���ڶ�ȡtfrecord�ļ�
�ļ��� tools�а����ű�train_crnn.py����ѵ�����磬�ű�mytest_crnn.py��test_crnn_jmz.py����ʶ��ͼƬ�е�����
�ļ��� test_imgs �а���������ͼƬ
�ļ��� model_save �б������ѵ���õ�ģ��

�˲�����Ҫ�Ļ�����Ϣ�� Requirement_Recognize_part.txt�С�
����ʱִ��test_crnn_jmz.py����Ŀ¼ BDCI_IDCARD��main_process.py���õ�ʾ����
    test_crnn_jmz.recognize_jmz(image_path=args.recognize_image_path, weights_path=args.recognize_weights_path,
              char_dict_path=args.recognize_char_dict_path, txt_file_path=args.recognize_txt_path)
����Ŀ¼BDCI_IDCARD�£����ն��е��õ�ʾ����
	python ./recognize_process/tools/test_crnn_jmz.py -i ./recognize_process/test_imgs/ -t ./recognize_process/anno_test
	����������Ϊ��
		(-i)image_path��     ��Ҫʶ���ͼƬ����·��     Ĭ���� ./recognize_process/test_imgs/
		(-w)weights_path:    ģ��Ȩ�ص�·��            Ĭ���� ./recognize_process/model_save/recognize_model
		(-c)char_dict_path:  �ֵ�����·��              Ĭ���� ./recognize_process/char_map/char_map.json
		(-t)txt_file_path:   ��ע�ļ�����·��          Ĭ����  ./recognize_process/anno_test/                   
	ģ�ͽ�����txt_file_path�ṩ��·������ѯ��Ŀ¼�µ�txt�ļ�����������txt�ļ��е�ͼƬ��������image_path�ṩ��ͼƬ·����ʶ�����ͼƬ��

ѵ��ʱִ��train_crnn.py�� ����Ŀ¼BDCI_IDCARD�£��ն˵��õ�ʾ����
	python ./recognize_process/tools/train_crnn.py -d ./data_tfrecord/ -s ./ckpt_save/
		����������Ϊ��
		(-d)dataset_dir��    ��Ҫʶ���ͼƬ����·��      Ĭ����  None
		(-w)weights_path:    Ԥѵ��ģ��Ȩ�ص�·��        Ĭ����  None
		(-c)char_dict_path:  �ֵ�����·��               Ĭ���� ./recognize_process/char_map/char_map.json
		(-s)save_path:       ��ע�ļ�����·��           Ĭ����  None                   
	����·��dataset_dir�µ�tfrecord����ѵ�����������weights_path��ΪNone�������ش�Ԥѵ��ģ�ͣ�ѵ�������ģ�ͽ�������save_path�¡�
	ѵ���еĸ�����������������Ŀ¼model�µ�model_config.py���޸ġ�һ����Ҫ�޸ĵĲ����У�
		__C.TRAIN.EPOCHS = 580000                          # ѵ����ֹ����
		__C.TRAIN.DISPLAY_STEP = 200                       # ѵ�������п��ӻ�����
		__C.TRAIN.LEARNING_RATE = 30000.0                  # ��ʼѧϰ��
		__C.TRAIN.BATCH_SIZE = 64                          # batch_size
		__C.TRAIN.LR_DECAY_STEPS = 2000                    # ʹ��ѧϰ��ָ��˥����˥������
		__C.TRAIN.LR_DECAY_RATE = 0.94                     # ˥��ֵ
		__C.TRAIN.SAVE_STEPS = 10000                       # ÿ�����ٲ�����һ��ģ��
		

ʶ��ģ�͵�ѵ�������������£�
	1.ʵ��Ӳ��������
		CPU��Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz��32��
		�ڴ棺128G��
		Swap������С��8G��
		GPU�� TITAN Xp �Դ�12G��
	2.ʵ�����������
		���Requirement_Recognize_part.txt
	3.ѵ�����̣�
		��1��.ʹ�ÿ�Դ���ݼ�ѵ��Լ9���step��batch sizeΪ128�� ѧϰ��Ϊ0.03�� ˥������Ϊ10000��˥��ֵΪ0.6�����ݼ����ӣ�
		https://pan.baidu.com/s/1ufYbnZAZ1q0AlK7yZ08cvQ   ��ע���ӣ� https://pan.baidu.com/s/1jfAKQVjD-SMJSffOwGhh8A�����룺u7bo
		
		��2��.ʹ�ÿ�Դ����https://github.com/Belval/TextRecognitionDataGenerator�������ֵ�char_map.json�е��ַ�������570���ų��Ȳ�һ���ַ�������4-19֮�䣩�������ȵ����ݡ�
		�ڣ�1���Ļ�����ѵ��������ѧϰ��Ϊ0.02��batch sizeΪ128��˥������Ϊ10000��˥��ֵΪ0.6��ѵ������Ϊ420000���������ܹ�ѵ����450000����
		
		��3��.�������֤������ģ���������֤ͼƬ�����ˮӡ��ȥ������Ϊѵ�����ݣ�������140����ѵ�����ݡ���ʼ��ѧϰ��Ϊ0.015��batch sizeΪ128��˥������Ϊ10000��˥��ֵΪ0.6��ѵ������Ϊ80000���������ܹ�ѵ����530000����
		
		��4��.ʹ�ó�����1����ѵ�����͸�����8000��ѵ��������ֳ�180000��ѵ��ͼƬ�����ڣ�3���Ļ����ϼ���ѵ������ʼ��ѧϰ��Ϊ0.010��batch sizeΪ128��˥������Ϊ1500��˥��ֵΪ0.94��ѵ������Ϊ90000���������ܹ�ѵ����620000����


