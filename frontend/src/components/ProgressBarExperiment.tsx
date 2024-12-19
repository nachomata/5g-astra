import { useEffect, useState } from "preact/hooks";
import { PUBLIC_BACKEND_URL } from "@/consts/const";
import type { StatusResponse } from "@/types/api";

export default function ProgressBarExperiment({ id, initialProgress } : { id: number, initialProgress: number }) {
    const fetchProgress = async () => {
        const res = await fetch(`${PUBLIC_BACKEND_URL}/simulation/${id}/status`);
        const { progress = 98 } = (await res.json()) as StatusResponse;
        setProgress(progress);
        if (progress === 100) window.location.reload();
    };

    const [progress, setProgress] = useState(initialProgress);

    useEffect(() => {
        const interval = setInterval(fetchProgress, 1000);
        return () => clearInterval(interval);
    }, [fetchProgress]);

    return (
        <div>
            <h1>Simulation in progress</h1>
            <p>Progress: {progress}%</p>
        </div>
    );

}