import { useState, useCallback, useEffect } from 'react';
import { WorkflowState } from '../../types';

export const useWorkflow = () => {
  const [myWorkflows, setMyWorkflows] = useState<WorkflowState[]>([]);
  const [workflow, setWorkflow] = useState<WorkflowState | null>(null);

  // Load workflows from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('omniflow_user_data');
    if (saved) {
      try {
        setMyWorkflows(JSON.parse(saved));
      } catch (e) {
        console.error('Failed to load workflows:', e);
      }
    }
  }, []);

  // Save workflow to localStorage
  const saveWorkflowToLocal = useCallback((current: WorkflowState) => {
    // Clean history to remove base64 data but keep URLs before saving to avoid localStorage quota issues
    const cleanedHistory = current.history.map(run => {
      const cleanedOutputs: Record<string, any> = {};
      // Keep URLs, but remove base64 data (data:image/..., data:video/..., data:audio/...)
      Object.entries(run.outputs || {}).forEach(([nodeId, output]) => {
        if (Array.isArray(output)) {
          cleanedOutputs[nodeId] = output.map((item: any) => {
            if (typeof item === 'string' && item.startsWith('data:')) {
              // Remove base64 data URLs
              return '';
            }
            return item; // Keep URLs (http/https) and other non-base64 data
          }).filter((item: any) => item !== '');
        } else if (typeof output === 'string') {
          if (output.startsWith('data:')) {
            // Remove base64 data URLs
            cleanedOutputs[nodeId] = '';
          } else {
            // Keep regular URLs (http/https)
            cleanedOutputs[nodeId] = output;
          }
        } else {
          cleanedOutputs[nodeId] = output;
        }
      });
      // Only keep outputs that have non-empty values
      Object.keys(cleanedOutputs).forEach(key => {
        if (cleanedOutputs[key] === '' || (Array.isArray(cleanedOutputs[key]) && cleanedOutputs[key].length === 0)) {
          delete cleanedOutputs[key];
        }
      });
      
      return {
        id: run.id,
        timestamp: run.timestamp,
        totalTime: run.totalTime,
        nodesSnapshot: run.nodesSnapshot,
        outputs: cleanedOutputs // Keep URLs, remove base64 data
      };
    });
    
    const updated = { 
      ...current, 
      updatedAt: Date.now(), 
      isDirty: false,
      history: cleanedHistory
    };
    
    setMyWorkflows(prev => {
      const next = prev.find(w => w.id === updated.id) 
        ? prev.map(w => w.id === updated.id ? updated : w) 
        : [updated, ...prev];
      
      try {
        localStorage.setItem('omniflow_user_data', JSON.stringify(next));
      } catch (e: any) {
        if (e.name === 'QuotaExceededError' || e.code === 22) {
          // If still too large, try to clean all workflows' history
          const fullyCleaned = next.map(w => ({
            ...w,
            history: w.history.map(run => {
              const cleanedOutputs: Record<string, any> = {};
              Object.entries(run.outputs || {}).forEach(([nodeId, output]) => {
                if (Array.isArray(output)) {
                  cleanedOutputs[nodeId] = output.map((item: any) => {
                    if (typeof item === 'string' && item.startsWith('data:')) {
                      return '';
                    }
                    return item;
                  }).filter((item: any) => item !== '');
                } else if (typeof output === 'string') {
                  if (!output.startsWith('data:')) {
                    cleanedOutputs[nodeId] = output;
                  }
                } else {
                  cleanedOutputs[nodeId] = output;
                }
              });
              Object.keys(cleanedOutputs).forEach(key => {
                if (cleanedOutputs[key] === '' || (Array.isArray(cleanedOutputs[key]) && cleanedOutputs[key].length === 0)) {
                  delete cleanedOutputs[key];
                }
              });
              return {
                id: run.id,
                timestamp: run.timestamp,
                totalTime: run.totalTime,
                nodesSnapshot: run.nodesSnapshot,
                outputs: cleanedOutputs
              };
            })
          }));
          try {
            localStorage.setItem('omniflow_user_data', JSON.stringify(fullyCleaned));
            return fullyCleaned;
          } catch (e2) {
            console.error('Failed to save workflows even after cleaning:', e2);
            return next;
          }
        } else {
          console.error('Failed to save workflows:', e);
          return next;
        }
      }
      return next;
    });
  }, []);

  // Delete workflow
  const deleteWorkflow = useCallback((id: string) => {
    setMyWorkflows(prev => {
      const next = prev.filter(w => w.id !== id);
      try {
        localStorage.setItem('omniflow_user_data', JSON.stringify(next));
      } catch (e) {
        console.error('Failed to delete workflow:', e);
      }
      return next;
    });
  }, []);

  return {
    myWorkflows,
    workflow,
    setWorkflow,
    setMyWorkflows,
    saveWorkflowToLocal,
    deleteWorkflow
  };
};


