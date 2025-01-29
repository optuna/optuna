# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: api.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'api.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\tapi.proto\x12\x06optuna\"W\n\x15\x43reateNewStudyRequest\x12*\n\ndirections\x18\x01 \x03(\x0e\x32\x16.optuna.StudyDirection\x12\x12\n\nstudy_name\x18\x02 \x01(\t\"\'\n\x13\x43reateNewStudyReply\x12\x10\n\x08study_id\x18\x01 \x01(\x03\"&\n\x12\x44\x65leteStudyRequest\x12\x10\n\x08study_id\x18\x01 \x01(\x03\"\x12\n\x10\x44\x65leteStudyReply\"L\n\x1cSetStudyUserAttributeRequest\x12\x10\n\x08study_id\x18\x01 \x01(\x03\x12\x0b\n\x03key\x18\x02 \x01(\t\x12\r\n\x05value\x18\x03 \x01(\t\"\x1c\n\x1aSetStudyUserAttributeReply\"N\n\x1eSetStudySystemAttributeRequest\x12\x10\n\x08study_id\x18\x01 \x01(\x03\x12\x0b\n\x03key\x18\x02 \x01(\t\x12\r\n\x05value\x18\x03 \x01(\t\"\x1e\n\x1cSetStudySystemAttributeReply\"/\n\x19GetStudyIdFromNameRequest\x12\x12\n\nstudy_name\x18\x01 \x01(\t\"+\n\x17GetStudyIdFromNameReply\x12\x10\n\x08study_id\x18\x01 \x01(\x03\"-\n\x19GetStudyNameFromIdRequest\x12\x10\n\x08study_id\x18\x01 \x01(\x03\"-\n\x17GetStudyNameFromIdReply\x12\x12\n\nstudy_name\x18\x01 \x01(\t\"-\n\x19GetStudyDirectionsRequest\x12\x10\n\x08study_id\x18\x01 \x01(\x03\"E\n\x17GetStudyDirectionsReply\x12*\n\ndirections\x18\x01 \x03(\x0e\x32\x16.optuna.StudyDirection\"1\n\x1dGetStudyUserAttributesRequest\x12\x10\n\x08study_id\x18\x01 \x01(\x03\"\xa6\x01\n\x1bGetStudyUserAttributesReply\x12P\n\x0fuser_attributes\x18\x01 \x03(\x0b\x32\x37.optuna.GetStudyUserAttributesReply.UserAttributesEntry\x1a\x35\n\x13UserAttributesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"3\n\x1fGetStudySystemAttributesRequest\x12\x10\n\x08study_id\x18\x01 \x01(\x03\"\xb0\x01\n\x1dGetStudySystemAttributesReply\x12V\n\x11system_attributes\x18\x01 \x03(\x0b\x32;.optuna.GetStudySystemAttributesReply.SystemAttributesEntry\x1a\x37\n\x15SystemAttributesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x16\n\x14GetAllStudiesRequest\"4\n\x12GetAllStudiesReply\x12\x1e\n\x07studies\x18\x01 \x03(\x0b\x32\r.optuna.Study\"p\n\x15\x43reateNewTrialRequest\x12\x10\n\x08study_id\x18\x01 \x01(\x03\x12%\n\x0etemplate_trial\x18\x02 \x01(\x0b\x32\r.optuna.Trial\x12\x1e\n\x16template_trial_is_none\x18\x03 \x01(\x08\"\'\n\x13\x43reateNewTrialReply\x12\x10\n\x08trial_id\x18\x01 \x01(\x03\"t\n\x18SetTrialParameterRequest\x12\x10\n\x08trial_id\x18\x01 \x01(\x03\x12\x12\n\nparam_name\x18\x02 \x01(\t\x12\x1c\n\x14param_value_internal\x18\x03 \x01(\x01\x12\x14\n\x0c\x64istribution\x18\x04 \x01(\t\"\x18\n\x16SetTrialParameterReply\"Q\n\'GetTrialIdFromStudyIdTrialNumberRequest\x12\x10\n\x08study_id\x18\x01 \x01(\x03\x12\x14\n\x0ctrial_number\x18\x02 \x01(\x03\"9\n%GetTrialIdFromStudyIdTrialNumberReply\x12\x10\n\x08trial_id\x18\x01 \x01(\x03\"a\n\x1aSetTrialStateValuesRequest\x12\x10\n\x08trial_id\x18\x01 \x01(\x03\x12!\n\x05state\x18\x02 \x01(\x0e\x32\x12.optuna.TrialState\x12\x0e\n\x06values\x18\x03 \x03(\x01\"1\n\x18SetTrialStateValuesReply\x12\x15\n\rtrial_updated\x18\x01 \x01(\x08\"^\n SetTrialIntermediateValueRequest\x12\x10\n\x08trial_id\x18\x01 \x01(\x03\x12\x0c\n\x04step\x18\x02 \x01(\x03\x12\x1a\n\x12intermediate_value\x18\x03 \x01(\x01\" \n\x1eSetTrialIntermediateValueReply\"L\n\x1cSetTrialUserAttributeRequest\x12\x10\n\x08trial_id\x18\x01 \x01(\x03\x12\x0b\n\x03key\x18\x02 \x01(\t\x12\r\n\x05value\x18\x03 \x01(\t\"\x1c\n\x1aSetTrialUserAttributeReply\"N\n\x1eSetTrialSystemAttributeRequest\x12\x10\n\x08trial_id\x18\x01 \x01(\x03\x12\x0b\n\x03key\x18\x02 \x01(\t\x12\r\n\x05value\x18\x03 \x01(\t\"\x1e\n\x1cSetTrialSystemAttributeReply\"#\n\x0fGetTrialRequest\x12\x10\n\x08trial_id\x18\x01 \x01(\x03\"-\n\rGetTrialReply\x12\x1c\n\x05trial\x18\x01 \x01(\x0b\x32\r.optuna.Trial\"_\n\x10GetTrialsRequest\x12\x10\n\x08study_id\x18\x01 \x01(\x03\x12\x1a\n\x12included_trial_ids\x18\x02 \x03(\x03\x12\x1d\n\x15trial_id_greater_than\x18\x03 \x01(\x03\"/\n\x0eGetTrialsReply\x12\x1d\n\x06trials\x18\x01 \x03(\x0b\x32\r.optuna.Trial\"\xc5\x02\n\x05Study\x12\x10\n\x08study_id\x18\x01 \x01(\x03\x12\x12\n\nstudy_name\x18\x02 \x01(\t\x12*\n\ndirections\x18\x03 \x03(\x0e\x32\x16.optuna.StudyDirection\x12:\n\x0fuser_attributes\x18\x04 \x03(\x0b\x32!.optuna.Study.UserAttributesEntry\x12>\n\x11system_attributes\x18\x05 \x03(\x0b\x32#.optuna.Study.SystemAttributesEntry\x1a\x35\n\x13UserAttributesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x37\n\x15SystemAttributesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xc3\x05\n\x05Trial\x12\x10\n\x08trial_id\x18\x01 \x01(\x03\x12\x0e\n\x06number\x18\x02 \x01(\x03\x12!\n\x05state\x18\x03 \x01(\x0e\x32\x12.optuna.TrialState\x12\x0e\n\x06values\x18\x04 \x03(\x01\x12\x16\n\x0e\x64\x61tetime_start\x18\x05 \x01(\t\x12\x19\n\x11\x64\x61tetime_complete\x18\x06 \x01(\t\x12)\n\x06params\x18\x07 \x03(\x0b\x32\x19.optuna.Trial.ParamsEntry\x12\x37\n\rdistributions\x18\x08 \x03(\x0b\x32 .optuna.Trial.DistributionsEntry\x12:\n\x0fuser_attributes\x18\t \x03(\x0b\x32!.optuna.Trial.UserAttributesEntry\x12>\n\x11system_attributes\x18\n \x03(\x0b\x32#.optuna.Trial.SystemAttributesEntry\x12\x42\n\x13intermediate_values\x18\x0b \x03(\x0b\x32%.optuna.Trial.IntermediateValuesEntry\x1a-\n\x0bParamsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01:\x02\x38\x01\x1a\x34\n\x12\x44istributionsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x35\n\x13UserAttributesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x37\n\x15SystemAttributesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x39\n\x17IntermediateValuesEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\r\n\x05value\x18\x02 \x01(\x01:\x02\x38\x01*,\n\x0eStudyDirection\x12\x0c\n\x08MINIMIZE\x10\x00\x12\x0c\n\x08MAXIMIZE\x10\x01*J\n\nTrialState\x12\x0b\n\x07RUNNING\x10\x00\x12\x0c\n\x08\x43OMPLETE\x10\x01\x12\n\n\x06PRUNED\x10\x02\x12\x08\n\x04\x46\x41IL\x10\x03\x12\x0b\n\x07WAITING\x10\x04\x32\xd7\r\n\x0eStorageService\x12L\n\x0e\x43reateNewStudy\x12\x1d.optuna.CreateNewStudyRequest\x1a\x1b.optuna.CreateNewStudyReply\x12\x43\n\x0b\x44\x65leteStudy\x12\x1a.optuna.DeleteStudyRequest\x1a\x18.optuna.DeleteStudyReply\x12\x61\n\x15SetStudyUserAttribute\x12$.optuna.SetStudyUserAttributeRequest\x1a\".optuna.SetStudyUserAttributeReply\x12g\n\x17SetStudySystemAttribute\x12&.optuna.SetStudySystemAttributeRequest\x1a$.optuna.SetStudySystemAttributeReply\x12X\n\x12GetStudyIdFromName\x12!.optuna.GetStudyIdFromNameRequest\x1a\x1f.optuna.GetStudyIdFromNameReply\x12X\n\x12GetStudyNameFromId\x12!.optuna.GetStudyNameFromIdRequest\x1a\x1f.optuna.GetStudyNameFromIdReply\x12X\n\x12GetStudyDirections\x12!.optuna.GetStudyDirectionsRequest\x1a\x1f.optuna.GetStudyDirectionsReply\x12\x64\n\x16GetStudyUserAttributes\x12%.optuna.GetStudyUserAttributesRequest\x1a#.optuna.GetStudyUserAttributesReply\x12j\n\x18GetStudySystemAttributes\x12\'.optuna.GetStudySystemAttributesRequest\x1a%.optuna.GetStudySystemAttributesReply\x12I\n\rGetAllStudies\x12\x1c.optuna.GetAllStudiesRequest\x1a\x1a.optuna.GetAllStudiesReply\x12L\n\x0e\x43reateNewTrial\x12\x1d.optuna.CreateNewTrialRequest\x1a\x1b.optuna.CreateNewTrialReply\x12U\n\x11SetTrialParameter\x12 .optuna.SetTrialParameterRequest\x1a\x1e.optuna.SetTrialParameterReply\x12\x82\x01\n GetTrialIdFromStudyIdTrialNumber\x12/.optuna.GetTrialIdFromStudyIdTrialNumberRequest\x1a-.optuna.GetTrialIdFromStudyIdTrialNumberReply\x12[\n\x13SetTrialStateValues\x12\".optuna.SetTrialStateValuesRequest\x1a .optuna.SetTrialStateValuesReply\x12m\n\x19SetTrialIntermediateValue\x12(.optuna.SetTrialIntermediateValueRequest\x1a&.optuna.SetTrialIntermediateValueReply\x12\x61\n\x15SetTrialUserAttribute\x12$.optuna.SetTrialUserAttributeRequest\x1a\".optuna.SetTrialUserAttributeReply\x12g\n\x17SetTrialSystemAttribute\x12&.optuna.SetTrialSystemAttributeRequest\x1a$.optuna.SetTrialSystemAttributeReply\x12:\n\x08GetTrial\x12\x17.optuna.GetTrialRequest\x1a\x15.optuna.GetTrialReply\x12=\n\tGetTrials\x12\x18.optuna.GetTrialsRequest\x1a\x16.optuna.GetTrialsReplyb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'api_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_GETSTUDYUSERATTRIBUTESREPLY_USERATTRIBUTESENTRY']._loaded_options = None
  _globals['_GETSTUDYUSERATTRIBUTESREPLY_USERATTRIBUTESENTRY']._serialized_options = b'8\001'
  _globals['_GETSTUDYSYSTEMATTRIBUTESREPLY_SYSTEMATTRIBUTESENTRY']._loaded_options = None
  _globals['_GETSTUDYSYSTEMATTRIBUTESREPLY_SYSTEMATTRIBUTESENTRY']._serialized_options = b'8\001'
  _globals['_STUDY_USERATTRIBUTESENTRY']._loaded_options = None
  _globals['_STUDY_USERATTRIBUTESENTRY']._serialized_options = b'8\001'
  _globals['_STUDY_SYSTEMATTRIBUTESENTRY']._loaded_options = None
  _globals['_STUDY_SYSTEMATTRIBUTESENTRY']._serialized_options = b'8\001'
  _globals['_TRIAL_PARAMSENTRY']._loaded_options = None
  _globals['_TRIAL_PARAMSENTRY']._serialized_options = b'8\001'
  _globals['_TRIAL_DISTRIBUTIONSENTRY']._loaded_options = None
  _globals['_TRIAL_DISTRIBUTIONSENTRY']._serialized_options = b'8\001'
  _globals['_TRIAL_USERATTRIBUTESENTRY']._loaded_options = None
  _globals['_TRIAL_USERATTRIBUTESENTRY']._serialized_options = b'8\001'
  _globals['_TRIAL_SYSTEMATTRIBUTESENTRY']._loaded_options = None
  _globals['_TRIAL_SYSTEMATTRIBUTESENTRY']._serialized_options = b'8\001'
  _globals['_TRIAL_INTERMEDIATEVALUESENTRY']._loaded_options = None
  _globals['_TRIAL_INTERMEDIATEVALUESENTRY']._serialized_options = b'8\001'
  _globals['_STUDYDIRECTION']._serialized_start=3476
  _globals['_STUDYDIRECTION']._serialized_end=3520
  _globals['_TRIALSTATE']._serialized_start=3522
  _globals['_TRIALSTATE']._serialized_end=3596
  _globals['_CREATENEWSTUDYREQUEST']._serialized_start=21
  _globals['_CREATENEWSTUDYREQUEST']._serialized_end=108
  _globals['_CREATENEWSTUDYREPLY']._serialized_start=110
  _globals['_CREATENEWSTUDYREPLY']._serialized_end=149
  _globals['_DELETESTUDYREQUEST']._serialized_start=151
  _globals['_DELETESTUDYREQUEST']._serialized_end=189
  _globals['_DELETESTUDYREPLY']._serialized_start=191
  _globals['_DELETESTUDYREPLY']._serialized_end=209
  _globals['_SETSTUDYUSERATTRIBUTEREQUEST']._serialized_start=211
  _globals['_SETSTUDYUSERATTRIBUTEREQUEST']._serialized_end=287
  _globals['_SETSTUDYUSERATTRIBUTEREPLY']._serialized_start=289
  _globals['_SETSTUDYUSERATTRIBUTEREPLY']._serialized_end=317
  _globals['_SETSTUDYSYSTEMATTRIBUTEREQUEST']._serialized_start=319
  _globals['_SETSTUDYSYSTEMATTRIBUTEREQUEST']._serialized_end=397
  _globals['_SETSTUDYSYSTEMATTRIBUTEREPLY']._serialized_start=399
  _globals['_SETSTUDYSYSTEMATTRIBUTEREPLY']._serialized_end=429
  _globals['_GETSTUDYIDFROMNAMEREQUEST']._serialized_start=431
  _globals['_GETSTUDYIDFROMNAMEREQUEST']._serialized_end=478
  _globals['_GETSTUDYIDFROMNAMEREPLY']._serialized_start=480
  _globals['_GETSTUDYIDFROMNAMEREPLY']._serialized_end=523
  _globals['_GETSTUDYNAMEFROMIDREQUEST']._serialized_start=525
  _globals['_GETSTUDYNAMEFROMIDREQUEST']._serialized_end=570
  _globals['_GETSTUDYNAMEFROMIDREPLY']._serialized_start=572
  _globals['_GETSTUDYNAMEFROMIDREPLY']._serialized_end=617
  _globals['_GETSTUDYDIRECTIONSREQUEST']._serialized_start=619
  _globals['_GETSTUDYDIRECTIONSREQUEST']._serialized_end=664
  _globals['_GETSTUDYDIRECTIONSREPLY']._serialized_start=666
  _globals['_GETSTUDYDIRECTIONSREPLY']._serialized_end=735
  _globals['_GETSTUDYUSERATTRIBUTESREQUEST']._serialized_start=737
  _globals['_GETSTUDYUSERATTRIBUTESREQUEST']._serialized_end=786
  _globals['_GETSTUDYUSERATTRIBUTESREPLY']._serialized_start=789
  _globals['_GETSTUDYUSERATTRIBUTESREPLY']._serialized_end=955
  _globals['_GETSTUDYUSERATTRIBUTESREPLY_USERATTRIBUTESENTRY']._serialized_start=902
  _globals['_GETSTUDYUSERATTRIBUTESREPLY_USERATTRIBUTESENTRY']._serialized_end=955
  _globals['_GETSTUDYSYSTEMATTRIBUTESREQUEST']._serialized_start=957
  _globals['_GETSTUDYSYSTEMATTRIBUTESREQUEST']._serialized_end=1008
  _globals['_GETSTUDYSYSTEMATTRIBUTESREPLY']._serialized_start=1011
  _globals['_GETSTUDYSYSTEMATTRIBUTESREPLY']._serialized_end=1187
  _globals['_GETSTUDYSYSTEMATTRIBUTESREPLY_SYSTEMATTRIBUTESENTRY']._serialized_start=1132
  _globals['_GETSTUDYSYSTEMATTRIBUTESREPLY_SYSTEMATTRIBUTESENTRY']._serialized_end=1187
  _globals['_GETALLSTUDIESREQUEST']._serialized_start=1189
  _globals['_GETALLSTUDIESREQUEST']._serialized_end=1211
  _globals['_GETALLSTUDIESREPLY']._serialized_start=1213
  _globals['_GETALLSTUDIESREPLY']._serialized_end=1265
  _globals['_CREATENEWTRIALREQUEST']._serialized_start=1267
  _globals['_CREATENEWTRIALREQUEST']._serialized_end=1379
  _globals['_CREATENEWTRIALREPLY']._serialized_start=1381
  _globals['_CREATENEWTRIALREPLY']._serialized_end=1420
  _globals['_SETTRIALPARAMETERREQUEST']._serialized_start=1422
  _globals['_SETTRIALPARAMETERREQUEST']._serialized_end=1538
  _globals['_SETTRIALPARAMETERREPLY']._serialized_start=1540
  _globals['_SETTRIALPARAMETERREPLY']._serialized_end=1564
  _globals['_GETTRIALIDFROMSTUDYIDTRIALNUMBERREQUEST']._serialized_start=1566
  _globals['_GETTRIALIDFROMSTUDYIDTRIALNUMBERREQUEST']._serialized_end=1647
  _globals['_GETTRIALIDFROMSTUDYIDTRIALNUMBERREPLY']._serialized_start=1649
  _globals['_GETTRIALIDFROMSTUDYIDTRIALNUMBERREPLY']._serialized_end=1706
  _globals['_SETTRIALSTATEVALUESREQUEST']._serialized_start=1708
  _globals['_SETTRIALSTATEVALUESREQUEST']._serialized_end=1805
  _globals['_SETTRIALSTATEVALUESREPLY']._serialized_start=1807
  _globals['_SETTRIALSTATEVALUESREPLY']._serialized_end=1856
  _globals['_SETTRIALINTERMEDIATEVALUEREQUEST']._serialized_start=1858
  _globals['_SETTRIALINTERMEDIATEVALUEREQUEST']._serialized_end=1952
  _globals['_SETTRIALINTERMEDIATEVALUEREPLY']._serialized_start=1954
  _globals['_SETTRIALINTERMEDIATEVALUEREPLY']._serialized_end=1986
  _globals['_SETTRIALUSERATTRIBUTEREQUEST']._serialized_start=1988
  _globals['_SETTRIALUSERATTRIBUTEREQUEST']._serialized_end=2064
  _globals['_SETTRIALUSERATTRIBUTEREPLY']._serialized_start=2066
  _globals['_SETTRIALUSERATTRIBUTEREPLY']._serialized_end=2094
  _globals['_SETTRIALSYSTEMATTRIBUTEREQUEST']._serialized_start=2096
  _globals['_SETTRIALSYSTEMATTRIBUTEREQUEST']._serialized_end=2174
  _globals['_SETTRIALSYSTEMATTRIBUTEREPLY']._serialized_start=2176
  _globals['_SETTRIALSYSTEMATTRIBUTEREPLY']._serialized_end=2206
  _globals['_GETTRIALREQUEST']._serialized_start=2208
  _globals['_GETTRIALREQUEST']._serialized_end=2243
  _globals['_GETTRIALREPLY']._serialized_start=2245
  _globals['_GETTRIALREPLY']._serialized_end=2290
  _globals['_GETTRIALSREQUEST']._serialized_start=2292
  _globals['_GETTRIALSREQUEST']._serialized_end=2387
  _globals['_GETTRIALSREPLY']._serialized_start=2389
  _globals['_GETTRIALSREPLY']._serialized_end=2436
  _globals['_STUDY']._serialized_start=2439
  _globals['_STUDY']._serialized_end=2764
  _globals['_STUDY_USERATTRIBUTESENTRY']._serialized_start=902
  _globals['_STUDY_USERATTRIBUTESENTRY']._serialized_end=955
  _globals['_STUDY_SYSTEMATTRIBUTESENTRY']._serialized_start=1132
  _globals['_STUDY_SYSTEMATTRIBUTESENTRY']._serialized_end=1187
  _globals['_TRIAL']._serialized_start=2767
  _globals['_TRIAL']._serialized_end=3474
  _globals['_TRIAL_PARAMSENTRY']._serialized_start=3204
  _globals['_TRIAL_PARAMSENTRY']._serialized_end=3249
  _globals['_TRIAL_DISTRIBUTIONSENTRY']._serialized_start=3251
  _globals['_TRIAL_DISTRIBUTIONSENTRY']._serialized_end=3303
  _globals['_TRIAL_USERATTRIBUTESENTRY']._serialized_start=902
  _globals['_TRIAL_USERATTRIBUTESENTRY']._serialized_end=955
  _globals['_TRIAL_SYSTEMATTRIBUTESENTRY']._serialized_start=1132
  _globals['_TRIAL_SYSTEMATTRIBUTESENTRY']._serialized_end=1187
  _globals['_TRIAL_INTERMEDIATEVALUESENTRY']._serialized_start=3417
  _globals['_TRIAL_INTERMEDIATEVALUESENTRY']._serialized_end=3474
  _globals['_STORAGESERVICE']._serialized_start=3599
  _globals['_STORAGESERVICE']._serialized_end=5350
# @@protoc_insertion_point(module_scope)
