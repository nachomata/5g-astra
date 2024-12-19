import { useEffect, useState } from "preact/hooks";
import { PUBLIC_BACKEND_URL } from "@/consts/const";
import type { StatusResponse } from "@/types/api";

export default function ProgressBarExperiment({ id, initialProgress }: { id: number, initialProgress: number }) {


    const [progress, setProgress] = useState(initialProgress);

    useEffect(() => {
        const fetchProgress = async () => {
            const res = await fetch(`${PUBLIC_BACKEND_URL}/experiment/${id}/status`);
            const { progress = 98 } = (await res.json()) as StatusResponse;
            setProgress(progress);
            if (progress === 100) window.location.reload();
        };
        const interval = setInterval(fetchProgress, 1000);
        return () => clearInterval(interval);
    });

    return (
        <article className="relative m-2 grid grid-cols-1 items-center justify-center h-24 bg-gray-900 rounded-lg border-white/20 border-2 w-full">
            <div className="absolute bg-gradient-to-r justify-self-start from-purple-600 to-blue-600 h-full transition-all" style={`width: ${progress}%`} />
            <div className="text-3xl text-center text-transparent z-10 tabular-nums bg-gradient-to-r" style={`background: linear-gradient(to right, white ${progress}%, gray ${progress}%); background-clip: text;`}><strong>{progress.toFixed(2)}%</strong></div>
        </article>
    );

}