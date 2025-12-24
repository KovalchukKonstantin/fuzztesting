import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export interface TaxonomyNode {
    id: string;
    content: string;
    full_path_content: string;
    rubric_score: number | null;
    ucb_score: number | null;
    depth: number;
}

export interface TreeNode {
    id: string;
    content: string;
    depth: number;
    status: string;
    rubric_score: number | null;
    ucb_score: number | null;
    human_label?: boolean;
    created_iteration: number;
    children: TreeNode[];
}

export interface IterationResponse {
    nodes_for_grading: TaxonomyNode[];
    iteration_id: number;
}

export interface FeedbackItem {
    node_id: string;
    is_relevant: boolean;
}

export interface StatsResponse {
    total_nodes: number;
    total_rubrics: number;
    total_labeled: number;
    current_rubric_principles: string[];
}

export interface AutoResult {
    iteration: number;
    new_nodes: number;
    rubric_updated: boolean;
}

export interface RubricPrinciple {
    id: string;
    description: string;
    weight: number;
}

export interface Rubric {
    id: string;
    iteration: number;
    principles: RubricPrinciple[];
}

export interface NodePrincipleScore {
    principle_id: string;
    score: number;
    reasoning: string | null;
}

export interface NodeScore {
    node_id: string;
    content: string;
    full_path_content: string;
    aggregate_score: number;
    principle_scores: NodePrincipleScore[];
}


export interface ContractorTask {
    task_id: string;
    node_id: string;
    content: string;
    full_path_content: string;
}

export interface ContractorBatchResponse {
    tasks: ContractorTask[];
    product_description: string | null;
    project_id: string | null;
    message?: string | null;
}

export interface ContractorSubmitResponse {
    status: string;
    loop_triggered: boolean;
}

export const api = {
    // Initialize project
    init: async (projectId: string, description: string, numTopics: number = 3) => {
        const res = await axios.post(`${API_BASE_URL}/init`, {
            project_id: projectId,
            product_description: description,
            num_initial_topics: numTopics,
            reinitialize: true
        });
        return res.data;
    },

    // Get next iteration nodes
    getNextIteration: async (k: number = 5, m: number = 3) => {
        const res = await axios.post<IterationResponse>(`${API_BASE_URL}/iteration/next?k=${k}&m=${m}`);
        return res.data;
    },

    // Submit feedback
    submitFeedback: async (items: FeedbackItem[]) => {
        const res = await axios.post(`${API_BASE_URL}/iteration/feedback`, { items });
        return res.data;
    },

    // Run auto iteration
    runAuto: async (iterations: number, mode: 'random' | 'llm') => {
        const res = await axios.post<{ status: string, results: AutoResult[] }>(`${API_BASE_URL}/iteration/auto`, {
            iterations,
            mode
        });
        return res.data;
    },

    // Get stats
    getState: async () => {
        const res = await axios.get<StatsResponse>(`${API_BASE_URL}/state`);
        return res.data;
    },

    // Get full tree
    getTree: async () => {
        const res = await axios.get<TreeNode>(`${API_BASE_URL}/tree`);
        return res.data;
    },

    // Get all rubrics
    getRubrics: async () => {
        const res = await axios.get<Rubric[]>(`${API_BASE_URL}/rubrics`);
        return res.data;
    },

    // Get scores for a rubric
    getRubricScores: async (iteration: number) => {
        const res = await axios.get<NodeScore[]>(`${API_BASE_URL}/rubrics/${iteration}/scores`);
        return res.data;
    },

    // Contractor API
    contractor: {
        fillQueue: async (limit: number = 50) => {
            const res = await axios.post(`${API_BASE_URL}/contractor/queue/fill`, { limit });
            return res.data;
        },
        getNextBatch: async (contractorId: string = "anon") => {
            const res = await axios.get<ContractorBatchResponse>(`${API_BASE_URL}/contractor/next?contractor_id=${contractorId}`);
            return res.data;
        },
        submitGrade: async (taskId: string, isRelevant: boolean, contractorId: string = "anon") => {
            const res = await axios.post<ContractorSubmitResponse>(`${API_BASE_URL}/contractor/task/${taskId}/submit`, {
                is_relevant: isRelevant,
                contractor_id: contractorId
            });
            return res.data;
        },
        submitBatchGrades: async (submissions: { task_id: string, is_relevant: boolean }[], contractorId: string = "anon") => {
            const res = await axios.post<{ status: string, loop_triggered: boolean }>(`${API_BASE_URL}/contractor/batch/submit`, {
                submissions,
                contractor_id: contractorId
            });
            return res.data;
        }
    },

    // Project Management
    getProjects: async () => {
        const res = await axios.get<{ projects: { id: string, product_description: string, created_at: string }[], current_project_id: string }>(`${API_BASE_URL}/projects`);
        return res.data;
    },

    switchProject: async (projectId: string) => {
        const res = await axios.post(`${API_BASE_URL}/project/switch`, { project_id: projectId });
        return res.data;
    },

    deleteProject: async (projectId: string) => {
        const res = await axios.delete(`${API_BASE_URL}/project/${projectId}`);
        return res.data;
    }
};

