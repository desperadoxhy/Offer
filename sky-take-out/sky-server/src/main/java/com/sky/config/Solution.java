package com.sky.config;

import io.swagger.models.auth.In;

import java.util.*;

public class Solution {
    /**
     * 快排
     * @param array
     */

    public static void sort(int[] array) {
        if (array == null || array.length < 2) {
            return;
        }
        quickSort(array, 0, array.length - 1);
    }


    private static void quickSort(int[] array, int left, int right) {
        if (left >= right) {
            return;
        }
        int pivot = partition(array, left, right);
        quickSort(array, left, pivot - 1);
        quickSort(array, pivot + 1, right);
    }

    private static int partition(int[] array, int left, int right) {
        int pivot = array[left];
        while (left < right) {
            while (left < right && array[right] >= pivot) {
                right --;
            }

            while (left < right && array[left] < pivot) {
                left++;
            }

            if (left < right) {
                swap(array, left, right);
            }
        }
        return array[left];
    }




    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> hash = new HashMap<>();
        for (int i = 0; i < nums.length; i++ ){
            if (hash.containsKey(target - nums[i])) {
                int [] arr = {hash.get(target - nums[i]), i};
                return arr;
            }
            hash.put(nums[i], i);
        }
        throw new RuntimeException();
    }

    public List<List<String>> groupAnagrams(String[] strs) {
        //判断是否为空字符串数组
        if(strs == null || strs.length == 0){
            return new ArrayList();
        }
        //1.创建一个哈希表
        Map<String,List> map = new HashMap<String, List>();
        for (String s: strs) {
            //将字符串转化为字符数组
            char[] chars = s.toCharArray();
            //对字符数组按照字母顺序排序
            Arrays.sort(chars);
            //将排序后的字符串作为哈希表中的key值
            String key = String.valueOf(chars);
            //2.判读哈希表中是否有该key值
            if (!map.containsKey(key)){
                //若不存在，则为新的异位词语，在map中创建新的键值对
                map.put(key,new ArrayList());
            }
            //3.将该字符串放在对应key的list中
            map.get(key).add(s);
        }
        //返回map中所有键值对象构成的list
        return new ArrayList(map.values());
    }

    public int longestConsecutive(int[] nums) {
        int res = 0;
        Set<Integer> Set = new HashSet<>();
        for (int num : nums) {
            Set.add(num);
        }
        int len;
        for (int num: Set){
            if (!Set.contains(num - 1)) {
                len = 1;
                while (Set.contains(++num)) len++;
                System.out.println(len);
                res = Math.max(res, len);
            }
        }
        return res;

    }

    /**
     * 移动零
     * @param nums
     */
    public void moveZeroes(int[] nums) {
        int n = nums.length, left = 0, right = 0;

        while (right < n) {
            //当前元素!=0，就把其交换到左边，等于0的交换到右边
            if (nums[right] != 0) {
                swap(nums, left, right);
                left++;
            }
            right++;
        }
    }


    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int res = Math.min(height[left], height[right]) * (right - left);
        while (left < right) {
            if (height[left] < height[right]) {
                left++;
                int area = Math.min(height[left], height[right]) * (right - left);
                res = Math.max(res, area);
            }else {
                right--;
                int area = Math.min(height[left], height[right]) * (right - left);
                res = Math.max(res, area);
            }
        }

        return res;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        Set<List<Integer>> list = new HashSet<List<Integer>>();
        List<List<Integer>> res = new ArrayList<>();
        int len = nums.length;
        if (nums == null || len < 3) {
            res.addAll(list);
            return res;
        }
        Arrays.sort(nums);
        for (int i = 0; i < len; i++){
            if (nums[i] > 0) break;
            if (i > 0 && nums[i] == nums[i-1]) continue;
            int L = i + 1;
            int R = len - 1;
            while (L < R) {
                int sum = nums[i] + nums[L] + nums[R];
                if (sum == 0){
                    list.add (Arrays.asList(nums[i], nums[L], nums[R]));
                    L++;
                    R--;
                }
                else if (sum < 0) L++;
                else if (sum > 0) R--;
            }
        }

        res.addAll(list);
        return res;
    }


    public int trap(int[] height) {
        int ans = 0;
        int left = 0;
        int right = height.length -1;
        int leftMax = 0, rightMax = 0;
        while (left < right) {
            leftMax = Math.max(leftMax, height[left]);
            rightMax = Math.max(rightMax, height[right]);

        }
        if (height[left] < height[right]) {
            ans += leftMax - height[left];
            ++left;
        } else {
            ans += rightMax - height[right];
            --right;
        }
        return ans;

    }

    public int lengthOfLongestSubstring(String s) {
        if (s.length() == 1){
            return 1;
        }

        HashMap<Character, Integer> occ = new HashMap<>();
        int res = 0;
        int slow = -1;
        for (int fast = 0; fast < s.length(); fast++) {
            if (occ.containsKey(s.charAt(fast))){
                if (slow < occ.get(s.charAt(fast))){
                    slow = occ.get(s.charAt(fast));
                }
            }
            occ.put(s.charAt(fast),fast);
            res = Math.max(res, fast - slow);

        }

        return res;

    }

    public List<Integer> findAnagrams(String s, String p) {
        int n = s.length(), m = p.length();
        List<Integer> res = new ArrayList<>();
        if(n < m) return res;
        int[] pCnt = new int[26];
        int[] sCnt = new int[26];
        for(int i = 0; i < m; i++){
            pCnt[p.charAt(i) - 'a']++;
            sCnt[s.charAt(i) - 'a']++;
        }
        if(Arrays.equals(sCnt, pCnt)){
            res.add(0);
        }
        for(int i = m; i < n; i++){
            sCnt[s.charAt(i - m) - 'a']--;
            sCnt[s.charAt(i) - 'a']++;
            if(Arrays.equals(sCnt, pCnt)){
                res.add(i - m + 1);
            }
        }
        return res;
    }

    public int subarraySum(int[] nums, int k) {
        int count = 0;
        for (int start = 0; start < nums.length; ++start) {
            int sum = 0;
            for (int end = start; end >= 0; --end) {
                sum += nums[end];
                if (sum == k) {
                    count++;
                    break;
                }
            }
        }
        return count;
    }

    public int subarraySum2(int[] nums, int k) {
        int count = 0, pre = 0;
        HashMap < Integer, Integer > mp = new HashMap < > ();
        mp.put(0, 1);

        for (int i = 0; i < nums.length; i++) {
            pre += nums[i];
            if (mp.containsKey(pre - k)) {
                count += mp.get(pre - k);
            }
            mp.put(pre, mp.getOrDefault(pre, 0) + 1);
        }
        return count;
    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums.length == 0 || k == 0) return new int[0];

        int[] res = new int[nums.length -k + 1];
        Deque<Integer> deque = new LinkedList<>();
        for (int j=0, i = 1 - k; j < nums.length; i++, j++){
            // 删除 deque 中对应的 nums[i-1]
            if(i > 0 && deque.peekFirst() == nums[i - 1])
                deque.removeFirst();
            // 保持 deque 递减
            while(!deque.isEmpty() && deque.peekLast() < nums[j])
                deque.removeLast();
            deque.addLast(nums[j]);
            // 记录窗口最大值
            if(i >= 0)
                res[i] = deque.peekFirst();
        }



        return res;
    }


    public String minWindow(String s, String t) {
        HashMap<Character,Integer> hs = new HashMap<Character,Integer>();
        HashMap<Character,Integer> ht = new HashMap<Character,Integer>();
        for(int i = 0;i < t.length();i ++){
            ht.put(t.charAt(i),ht.getOrDefault(t.charAt(i), 0) + 1);
        }
        String ans = "";
        int len = 1000000, cnt = 0;  //有多少个元素符合
        for(int i = 0,j = 0;i < s.length();i ++) {
            hs.put(s.charAt(i), hs.getOrDefault(s.charAt(i), 0) + 1);
            if(ht.containsKey(s.charAt(i)) && hs.get(s.charAt(i)) <= ht.get(s.charAt(i))) cnt ++;
            while(j < i && (!ht.containsKey(s.charAt(j)) || hs.get(s.charAt(j)) > ht.get(s.charAt(j))))
            {
                int count = hs.get(s.charAt(j)) - 1;
                hs.put(s.charAt(j), count);
                j ++;
            }
            if(cnt == t.length() && i - j + 1 < len){
                len = i - j + 1;
                ans = s.substring(j,i + 1);
            }
        }
        return ans;
    }

    public int maxSubArray(int[] nums) {
        int pre = 0;
        int res = nums[0];
        for (int num : nums) {
            pre = Math.max(pre + num, num);
            res = Math.max(res, pre);
        }
        return res;
    }

    public int[][] merge(int[][] intervals) {
        // 先按照区间起始位置排序
        Arrays.sort(intervals, (v1, v2) -> v1[0] - v2[0]);
        // 遍历区间
        int[][] res = new int[intervals.length][2];
        int idx = -1;
        for (int[] interval: intervals) {
            // 如果结果数组是空的，或者当前区间的起始位置 > 结果数组中最后区间的终止位置，
            // 则不合并，直接将当前区间加入结果数组。
            if (idx == -1 || interval[0] > res[idx][1]) {
                res[++idx] = interval;
            } else {
                // 反之将当前区间合并至结果数组的最后区间
                res[idx][1] = Math.max(res[idx][1], interval[1]);
            }
        }
        return Arrays.copyOf(res, idx + 1);
    }


















    public static void swap(int[] nums, int left, int right) {
        int temp = nums[left];
        nums[left] = nums[right];
        nums[right] = temp;
    }




    public static void main(String[] args) {
        Solution s = new Solution();
        int[] arr = {1,3,-1,-3,5,3,6,7};
        int[] res = s.maxSlidingWindow(arr,3);
        System.out.println(res);
    }
}
