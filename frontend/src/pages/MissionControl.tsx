import { SkeletonLoader } from '../components/common/SkeletonLoader';

// In render:
{isLoading && <SkeletonLoader />}
{isError && <div className="text-red-500">Error loading data</div>}
