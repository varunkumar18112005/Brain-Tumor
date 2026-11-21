import axios from 'axios';
import {
  UploadResponse,
  PreprocessingResponse,
  SegmentationResponse,
  ExtractedFeatures,
  FullAnalysisResult,
  AnalysisHistoryItem,
  ModelMetrics,
  ModelStatus
} from '../types';

const api = axios.create({
  baseURL: '/api'
});

export const uploadImage = async (file: File, modality: string, patientId: string): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('modality', modality);
  formData.append('patient_id', patientId);

  const res = await api.post<UploadResponse>('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return res.data;
};

export const preprocessScan = async (fileId: string, modality: string): Promise<PreprocessingResponse> => {
  const res = await api.post<PreprocessingResponse>('/preprocess', {
    file_id: fileId,
    modality: modality
  });
  return res.data;
};

export const segmentScan = async (fileId: string, modality: string, useModel: boolean = true): Promise<SegmentationResponse> => {
  const res = await api.post<SegmentationResponse>('/segment', {
    file_id: fileId,
    modality: modality,
    use_model: useModel
  });
  return res.data;
};

export const extractFeatures = async (fileId: string): Promise<ExtractedFeatures> => {
  const res = await api.post<ExtractedFeatures>('/extract-features', {
    file_id: fileId
  });
  return res.data;
};

export const runFullAnalysis = async (
  fileId: string,
  patientId: string,
  modality: string,
  selectedSlice: number = 0
): Promise<FullAnalysisResult> => {
  const res = await api.post<FullAnalysisResult>('/analyze', {
    file_id: fileId,
    patient_id: patientId,
    modality: modality,
    selected_slice: selectedSlice
  });
  return res.data;
};

export const getHistory = async (modality?: string, tumorStatus?: string): Promise<AnalysisHistoryItem[]> => {
  const params: Record<string, string> = {};
  if (modality && modality !== 'ALL') params.modality = modality;
  if (tumorStatus && tumorStatus !== 'ALL') params.tumor_status = tumorStatus;

  const res = await api.get<AnalysisHistoryItem[]>('/history', { params });
  return res.data;
};

export const getAnalysisById = async (id: number): Promise<FullAnalysisResult> => {
  const res = await api.get<FullAnalysisResult>(`/history/${id}`);
  return res.data;
};

export const getModelStatus = async (): Promise<ModelStatus> => {
  const res = await api.get<ModelStatus>('/model/status');
  return res.data;
};

export const getModelMetrics = async (): Promise<ModelMetrics> => {
  const res = await api.get<ModelMetrics>('/model/metrics');
  return res.data;
};

export const trainModel = async (epochs: number = 5, modality: string = 'MRI'): Promise<any> => {
  const res = await api.post('/model/train', null, {
    params: { epochs, modality }
  });
  return res.data;
};

export const getKnowledgeBase = async (): Promise<any> => {
  const res = await api.get('/knowledge-base');
  return res.data;
};

export const updateKnowledgeBase = async (payload: any): Promise<any> => {
  const res = await api.post('/knowledge-base/rules', payload);
  return res.data;
};

export const getFuzzyRules = async (): Promise<any> => {
  const res = await api.get('/fuzzy/rules');
  return res.data;
};

export const updateFuzzyRules = async (payload: any): Promise<any> => {
  const res = await api.post('/fuzzy/rules', payload);
  return res.data;
};

export default api;
