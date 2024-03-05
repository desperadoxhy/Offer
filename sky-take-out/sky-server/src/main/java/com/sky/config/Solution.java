package com.sky.config;

import io.swagger.models.auth.In;
import org.apache.poi.ss.formula.functions.Intercept;
import org.apache.poi.ss.formula.functions.T;
import org.checkerframework.checker.units.qual.A;

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

    public int[] productExceptSelf(int[] nums) {
        int[] res = new int[nums.length];
        int[] L = new int[nums.length];
        int[] R = new int[nums.length];

        L[0] = 1;
        for (int i = 0; i < nums.length -1; i++) {
            L[i + 1] = nums[i] * L[i];
        }

        R[nums.length - 1] = 1;
        for (int i = nums.length - 1; i > 0; i--) {
            R[i - 1] = R[i] * nums[i];
        }

        // 对于索引 i，除 nums[i] 之外其余各元素的乘积就是左侧所有元素的乘积乘以右侧所有元素的乘积
        for (int i = 0; i < nums.length; i++) {
            res[i] = L[i] * R[i];
        }



        return res;

    }

    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                swap(nums, nums[i] - 1, i);
            }
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }

    public void setZeroes(int[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;
        boolean row0_flag = false;
        boolean col0_flag = false;
        // 第一行是否有零
        for (int j = 0; j < col; j++) {
            if (matrix[0][j] == 0) {
                row0_flag = true;
                break;
            }
        }
        // 第一列是否有零
        for (int i = 0; i < row; i++) {
            if (matrix[i][0] == 0) {
                col0_flag = true;
                break;
            }
        }
        // 把第一行第一列作为标志位
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        // 置0
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (row0_flag) {
            for (int j = 0; j < col; j++) {
                matrix[0][j] = 0;
            }
        }
        if (col0_flag) {
            for (int i = 0; i < row; i++) {
                matrix[i][0] = 0;
            }
        }
    }

    public List<Integer> spiralOrder(int[][] matrix) {
        if (matrix.length == 0)
            return new ArrayList<Integer>();
        int l = 0, r = matrix[0].length - 1, t = 0, b = matrix.length - 1, x = 0;
        Integer[] res = new Integer[(r + 1) * (b + 1)];
        while (true) {
            for (int i = l; i <= r; i++) res[x++] = matrix[t][i]; // left to right
            if (++t > b) break;
            for (int i = t; i <= b; i++) res[x++] = matrix[i][r]; // top to bottom
            if (l > --r) break;
            for (int i = r; i >= l; i--) res[x++] = matrix[b][i]; // right to left
            if (t > --b) break;
            for (int i = b; i >= t; i--) res[x++] = matrix[i][l]; // bottom to top
            if (++l > r) break;
        }
        return Arrays.asList(res);
    }

    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n / 2; ++i) {
            for (int j = 0; j < (n + 1) / 2; ++j) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - j - 1][i];
                matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
                matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
                matrix[j][n - i - 1] = temp;
            }
        }
    }


    public boolean searchMatrix(int[][] matrix, int target) {
        int i = matrix.length - 1, j = 0;
        while(i >= 0 && j < matrix[0].length)
        {
            if(matrix[i][j] > target) i--;
            else if(matrix[i][j] < target) j++;
            else return true;
        }
        return false;
    }


     public class ListNode {
         int val;
         ListNode next;
         ListNode() {}
         ListNode(int val) { this.val = val; }
         ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     }


    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode a = new ListNode();
        ListNode cur = a;

        while (list1 != null && list2 != null) {
            if (list1.val < list2.val) {
                cur.next = list1;
                list1 = list1.next;
                cur = cur.next;
            } else {
                cur.next = list1;
                list1 = list1.next;
                cur = cur.next;
            }
        }

        cur.next = list1 != null ? list1 : list2;
        return a.next;

    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode pre = new ListNode(0);
        ListNode cur = pre;
        int carry = 0;
        while(l1 != null || l2 != null) {
            int x = l1 == null ? 0 : l1.val;
            int y = l2 == null ? 0 : l2.val;
            int sum = x + y + carry;

            carry = sum / 10;
            sum = sum % 10;
            cur.next = new ListNode(sum);

            cur = cur.next;
            if(l1 != null)
                l1 = l1.next;
            if(l2 != null)
                l2 = l2.next;
        }
        if(carry == 1) {
            cur.next = new ListNode(carry);
        }
        return pre.next;
    }


    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode pre = new ListNode();
        pre.next = head;
        ListNode fast = pre;
        ListNode slow = pre;
        for (int i =0; i < n; i++) {
            fast = fast.next;
        }
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }

        slow.next = slow.next.next;

        return pre.next;
    }


    public ListNode swapPairs(ListNode head) {
        ListNode pre = new ListNode(0);
        pre.next = head;
        ListNode temp = pre;
        while(temp.next != null && temp.next.next != null) {
            ListNode start = temp.next;
            ListNode end = temp.next.next;
            temp.next = end;
            start.next = end.next;
            end.next = start;
            temp = start;
        }
        return pre.next;
    }

    /**
     * 递归解法
     * @param head
     * @return
     */
    public ListNode swapPairs2(ListNode head) {
        if(head == null || head.next == null){
            return head;
        }

        ListNode next = head.next;
        head.next = swapPairs(next.next);
        next.next = head;
        return next;

    }
























    public static void swap(int[] nums, int left, int right) {
        int temp = nums[left];
        nums[left] = nums[right];
        nums[right] = temp;
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode pre = new ListNode(0);
        pre.next = head;

        ListNode begin = pre;
        ListNode end = pre;

        while (end.next != null) {
            for (int i = 0; i < k && end != null; i++) {
                end = end.next;
            }
            if (end == null) {
                break;
            }
            ListNode start = begin.next;
            ListNode next = end.next;
            end.next = null;
            begin.next = reverse(start);
            start.next = next;
            end = begin;


        }

        return pre.next;

    }

    private ListNode reverse(ListNode head) {
        ListNode pre = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = pre;
            pre = curr;
            curr = next;
        }
        return pre;
    }

    public Node copyRandomList(Node head) {
        if(head == null) return null;
        Node cur = head;
        Map<Node, Node> map = new HashMap<>();
        // 3. 复制各节点，并建立 “原节点 -> 新节点” 的 Map 映射
        while(cur != null) {
            map.put(cur, new Node(cur.val));
            cur = cur.next;
        }
        cur = head;
        // 4. 构建新链表的 next 和 random 指向
        while(cur != null) {
            map.get(cur).next = map.get(cur.next);
            map.get(cur).random = map.get(cur.random);
            cur = cur.next;
        }
        // 5. 返回新链表的头节点
        return map.get(head);
    }

    public ListNode sortList(ListNode head) {
        return mergeSort(head);
    }

    /**
     * 对给定的链表进行归并排序
     */
    ListNode mergeSort(ListNode head){
        // 如果链表为空或只有一个节点，无需排序直接返回
        if(head == null || head.next == null){
            return head;
        }
        // 获取链表的中间节点，分别对左右子链表进行排序
        ListNode mid = getMid(head);
        ListNode rightSorted = mergeSort(mid.next);   // 排序右子链表
        mid.next = null;                     // 断开两段子链表
        ListNode leftSorted = mergeSort(head);         // 排序左子链表
        return mergeTwoLists2(leftSorted, rightSorted);  // 两个子链表必然有序，合并两个有序的链表
    }

    /**
     * 获取以head为头节点的链表中间节点
     * 如果链表长度为奇数，返回最中间的那个节点
     * 如果链表长度为偶数，返回中间靠左的那个节点
     */
    ListNode getMid(ListNode head){
        if(head == null)return head;
        ListNode slow = head, fast = head.next;          // 快慢指针，慢指针初始为
        while(fast != null && fast.next != null)
        {
            fast = fast.next.next;    // 快指针每次移动两个节点
            slow = slow.next;         // 慢指针每次移动一个节点
        }
        return slow;    // 快指针到达链表尾部时，慢指针即指向中间节点
    }

    /**
     * 合并两个有序链表list1和list2
     */
    ListNode mergeTwoLists2(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode();   // 伪头节点，用于定位合并链表的头节点
        ListNode node = dummy;             // 新链表当前的最后一个节点，初始为伪头节点
        // 直到两个链表都遍历完了，合并结束
        while(list1 != null || list2 != null){
            int val1 = list1 == null ? 50001 : list1.val;   // 如果链表1已经遍历完，val1取最大值，保证链表2的节点被选择到
            int val2 = list2 == null ? 50001 : list2.val;   // 如果链表2已经遍历完，val2取最大值，保证链表1的节点被选择到
            if(val1 < val2){
                // 链表1的节点值更小，加入到合并链表，并更新链表1指向的节点
                node.next = list1;
                list1 = list1.next;
            }else{
                // 链表2的节点值更小，加入到合并链表，并更新链表2指向的节点
                node.next = list2;
                list2 = list2.next;
            }
            node = node.next;    // 更新合并链表当前的最后一个节点指向
        }
        return dummy.next;       // 伪头节点的下一个节点即为合并链表的头节点
    }

    public ListNode mergeKLists(ListNode[] lists) {
        ListNode res = null;
        for (ListNode list: lists) {
            res = mergeTwoLists2(res, list);
        }
        return res;
    }


    class LRUCache extends LinkedHashMap<Integer, Integer>{

        private int capacity;

        public LRUCache(int capacity) {
            super(capacity, 0.75F, true);
            this.capacity = capacity;
        }

        public int get(int key) {
            return super.getOrDefault(key, -1);
        }

        // 这个可不写
        public void put(int key, int value) {
            super.put(key, value);
        }

        @Override
        protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
            return size() > capacity;
        }

    }


    /**
     * 二叉树的中序遍历
     * @param root
     * @return
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        inorder(root, res);
        return res;
    }

    public void inorder(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }
        inorder(root.left, res);
        res.add(root.val);
        inorder(root.right, res);
    }

    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }



    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }

    public boolean isSymmetric(TreeNode root) {
        return check(root, root);
    }

    public boolean check(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        return p.val == q.val && check(p.left, q.right) && check(p.right, q.left);
    }

    int ans;
    public int diameterOfBinaryTree(TreeNode root) {
        ans = 1;
        depth(root);
        return ans - 1;
    }
    public int depth(TreeNode node) {
        if (node == null) {
            return 0; // 访问到空节点了，返回0
        }
        int L = depth(node.left); // 左儿子为根的子树的深度
        int R = depth(node.right); // 右儿子为根的子树的深度
        ans = Math.max(ans, L+R+1); // 计算d_node即L+R+1 并更新ans
        return Math.max(L, R) + 1; // 返回该节点为根的子树的深度
    }

    /**
     * 二叉树的层序遍历
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        if (root == null) {
            return ret;
        }

        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            List<Integer> level = new ArrayList<>();
            int size = queue.size();
            for (int i =1; i <= size; i++){
                TreeNode node = queue.poll();
                level.add(node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            ret.add(level);


        }
        return ret;

    }


    public TreeNode sortedArrayToBST(int[] nums) {
        return helper(nums, 0, nums.length - 1);
    }

    public TreeNode helper(int[] nums, int left, int right) {
        if (left > right) {
            return null;
        }

        // 总是选择中间位置左边的数字作为根节点
        int mid = (left + right) / 2;

        TreeNode root = new TreeNode(nums[mid]);
        root.left = helper(nums, left, mid - 1);
        root.right = helper(nums, mid + 1, right);
        return root;
    }

    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    public boolean isValidBST(TreeNode node, long lower, long upper) {
        if (node == null) {
            return true;
        }
        if (node.val <= lower || node.val >= upper) {
            return false;
        }
        return isValidBST(node.left, lower, node.val) && isValidBST(node.right, node.val, upper);
    }

    int res, k;
    void dfs(TreeNode root) {
        if (root == null) return;
        dfs(root.left);
        if (k == 0) return;
        if (--k == 0) res = root.val;
        dfs(root.right);
    }
    public int kthSmallest(TreeNode root, int k) {
        this.k = k;
        dfs(root);
        return res;
    }

    /**
     * 二叉树的右视图
     *使用层序遍历，最右边的节点为所求值
     * @param root
     * @return
     */
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int num = queue.size();
            for (int i = 0; i < num; i++) {
                TreeNode node = queue.poll();
                if (i == num - 1) {
                    res.add(node.val);
                }
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        return res;
    }


    /**
     * 二叉树转化为链表
     * 使用栈实现前序遍历
     * 前序遍历和节点转换同时进行
     * @param root
     */
    public void flatten(TreeNode root) {
        if (root == null) return;
        Deque<TreeNode> stack = new LinkedList<TreeNode>();
        stack.push(root);
        TreeNode pre = null;
        while (!stack.isEmpty()) {
            TreeNode cur = stack.pop();

            if (pre != null) {
                pre.left = null;
                pre.right = cur;
            }

            TreeNode left = cur.left, right = cur.right;

            if (right != null) {
                stack.push(right);
            }
            if (left != null) {
                stack.push(left);
            }

            pre = cur;
        }
    }

    /**
     * 从前序序列和中序序列构造二叉树
     * 采用递归的方法
     * @param preorder
     * @param inorder
     * @return
     */
    private Map<Integer, Integer> index;

    public TreeNode buildTree(int[] preorder, int[] inorder){
        int n = preorder.length;
        index = new HashMap<Integer, Integer>();
        for (int i = 0; i < n; i++) {
            index.put(inorder[i], i);
        }
        return myBuildTree(preorder, inorder, 0, n-1, 0, n-1);
    }

    TreeNode myBuildTree(int[] preorder, int[] inorder, int preorder_left, int preorder_right, int inorder_left, int inorder_right) {
        if (preorder_left > preorder_right) {
            return null;
        }
        int pre_root = preorder_left;
        int inorder_root = index.get(preorder[pre_root]);

        TreeNode root = new TreeNode(preorder[preorder_left]);

        int left_size = inorder_root - inorder_left;

        root.left = myBuildTree(preorder, inorder, preorder_left + 1, preorder_left + left_size, inorder_left, inorder_root - 1);

        root.right = myBuildTree(preorder, inorder, preorder_left + left_size + 1, preorder_right, inorder_root + 1, inorder_right);

        return root;
    }


    /**
     * 路径总和
     * 使用前缀加状态恢复
     * 两个节点之间的路径和 = 两个节点之前的前缀之差
     * 状态用于遍历玩一个节点的所有字节点后，将其从map中去除
     * @param root
     * @param targetSum
     * @return
     */
    Map<Long, Integer> prefixMap;
    int target;
    public int pathSum(TreeNode root, int targetSum) {
        prefixMap = new HashMap<>();
        target = targetSum;
        prefixMap.put(0L, 1); // 前缀为0的个数至少一个
        return recur(root, 0);
    }

    int recur(TreeNode node, long curSum) {
        if (node == null) {
            return 0;
        }

        int res = 0;
        curSum += node.val; // 得到当前前缀树的值

        res += prefixMap.getOrDefault(curSum - target, 0); //得到需要的前缀树的个数；
        prefixMap.put(curSum, prefixMap.getOrDefault(curSum, 0) + 1);
        int left = recur(node.left, curSum);
        int right = recur(node.right, curSum);
        prefixMap.put(curSum, prefixMap.get(curSum) - 1);
        return res + left + right;

    }

    /**
     * 寻找两个子节点的最近公共祖先
     *
     * 递归写法，分别去左边和右边找
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;

        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        //p和q都没找到，那就没有
        if(left == null && right == null) {
            return null;
        }

        //左子树没有p也没有q，就返回右子树的结果
        if(left == null) return right;

        //右子树没有p也没有q就返回左子树的结果
        if(right == null) return left;

        //左右子树都找到p和q了，那就说明p和q分别在左右两个子树上，所以此时的最近公共祖先就是root
        return root;

    }

    /**
     * 二叉树中的最大路径和
     * 仅使用当前节点、使用当前节点和左子树路径 和 使用当前节点和右子树路径 三者中取最大值进行返回。
     * 当左右节点路径和不为负数时，说明能够对当前路径起到正向贡献作用，将其添加到路径中
     */
    int ans2 = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        dfs2(root);
        return ans2;
    }
    int dfs2(TreeNode root) {
        if (root == null) return 0;
        int left = dfs2(root.left), right = dfs2(root.right);
        int t = root.val;
        if (left >= 0) t += left;
        if (right >= 0) t += right;
        ans2 = Math.max(ans2, t);
        return Math.max(root.val, Math.max(left, right) + root.val);
    }

    /**
     * 岛屿数量
     * 每找到一个岛屿，将四周的值都变成0
     * @param grid
     * @return
     */
    public int numIslands(char[][] grid) {
        int count = 0;
        for (int i = 0; i < grid.length; i++){
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    dfs_area(grid, i, j);
                    count ++;
                }
            }
        }
        return count;
    }


    private void dfs_area(char[][] grid, int i, int j){
        if(i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] == '0') return;
        grid[i][j] = '0';
        dfs_area(grid, i + 1, j);
        dfs_area(grid, i, j + 1);
        dfs_area(grid, i - 1, j);
        dfs_area(grid, i, j - 1);
    }


    /**
     * 岛屿遍历工具
     * @param grid
     * @param r
     * @param c
     * @return
     */
    int area(int[][] grid, int r, int c) {
        if (!inArea(grid, r, c)) {
            return 0;
        }
        if (grid[r][c] != 1) {
            return 0;
        }
        grid[r][c] = 2;

         area(grid, r - 1, c);
         area(grid, r + 1, c);
         area(grid, r, c - 1);
         area(grid, r, c + 1);
         return 1;
    }

    boolean inArea(int[][] grid, int r, int c) {
        return 0 <= r && r < grid.length
                && 0 <= c && c < grid[0].length;
    }

    /**
     * 腐烂的橘子
     * BFS 可以用来求最短路径问题。BFS 先搜索到的结点，一定是距离最近的结点。
     *求腐烂橘子到所有新鲜橘子的最短路径
     *
     *在 BFS 中，每遍历到一个橘子（污染了一个橘子），就将新鲜橘子的数量减一。如果 BFS 结束后这个数量仍未减为零，说明存在无法被污染的橘子。
     * @param grid
     * @return
     */
    public int orangesRotting(int[][] grid) {
        int M = grid.length;
        int N = grid[0].length;
        Queue<int[]> queue = new LinkedList<>();

        int count = 0; // count 表示新鲜橘子的数量
        for (int r = 0; r < M; r++) {
            for (int c = 0; c < N; c++) {
                if (grid[r][c] == 1) {
                    count++;
                } else if (grid[r][c] == 2) {
                    queue.add(new int[]{r, c});
                }
            }
        }

        int round = 0; // round 表示腐烂的轮数，或者分钟数
        while (count > 0 && !queue.isEmpty()) {
            round++;
            int n = queue.size();
            for (int i = 0; i < n; i++) {
                int[] orange = queue.poll();
                int r = orange[0];
                int c = orange[1];
                if (r-1 >= 0 && grid[r-1][c] == 1) {
                    grid[r-1][c] = 2;
                    count--;
                    queue.add(new int[]{r-1, c});
                }
                if (r+1 < M && grid[r+1][c] == 1) {
                    grid[r+1][c] = 2;
                    count--;
                    queue.add(new int[]{r+1, c});
                }
                if (c-1 >= 0 && grid[r][c-1] == 1) {
                    grid[r][c-1] = 2;
                    count--;
                    queue.add(new int[]{r, c-1});
                }
                if (c+1 < N && grid[r][c+1] == 1) {
                    grid[r][c+1] = 2;
                    count--;
                    queue.add(new int[]{r, c+1});
                }
            }
        }

        if (count > 0) {
            return -1;
        } else {
            return round;
        }

    }

    /**
     * 课程表
     *  安排图是否是 有向无环图(DAG)
     *  拓扑排序原理： 对 DAG 的顶点进行排序，使得对每一条有向边 (u,v)(u, v)(u,v)，均有 uuu（在排序记录中）比 vvv 先出现。亦可理解为对某点 vvv 而言，只有当 vvv 的所有源点均出现了，vvv 才能出现。
     *
     * i == 0 ： 干净的，未被 DFS 访问
     * i == -1：其他节点启动的 DFS 访问过了，路径没问题，不需要再访问了
     * i == 1  ：本节点启动的 DFS 访问过了，一旦遇到了也说明有环了
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<List<Integer>> adj = new ArrayList<>();
        for (int i=0; i < numCourses; i++) {
            adj.add(new ArrayList<>());
        }
        int[] flags = new int[numCourses];

        for(int[] cp : prerequisites) adj.get(cp[1]).add(cp[0]);

        for(int i = 0; i < numCourses; i++) {
            if(!dfs_corse(adj, flags, i)) return false;
            return true;
        }


        return true;
    }

    boolean dfs_corse(List<List<Integer>> adj, int[] flags, int i) {
        if(flags[i] == 1) return false;
        if(flags[i] == -1) return true;
        flags[i] = 1;
        for(Integer j : adj.get(i))
            if(!dfs_corse(adj, flags, j)) return false;
        flags[i] = -1;
        return true;
    }

    /**
     * 爬楼梯
     * 递归加动态规划
     * 类似斐波那契数列
     *
     * 当为 1 级台阶： 剩 n−1 个台阶，此情况共有 f(n−1) 种跳法。
     * 当为 2 级台阶： 剩 n−2 个台阶，此情况共有 f(n−2)种跳法。
     *
     * f(n)=f(n−1)+f(n−2)
     * @param n
     * @return
     */
    public int climbStairs(int n) {
        int a = 1, b = 1, sum;
        for (int i = 0; i < n -1; i++) {
            sum = a + b;
            a = b;
            b = sum;
        }
        return b;
    }

    /**
     * 杨辉三角
     * @param numRows
     * @return
     */
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        for (int i = 0; i < numRows; ++i) {
            List<Integer> row = new ArrayList<Integer>();
            for (int j = 0; j <= i; ++j) {
                if (j == 0 || j == i) {
                    row.add(1);
                } else {
                    row.add(ret.get(i - 1).get(j - 1) + ret.get(i - 1).get(j));
                }
            }
            ret.add(row);
        }
        return ret;
    }

    /**
     * 打家劫舍
     * 定义动态规划问题的几个步骤
     * 1 定义子问题
     * 2 写出子问题的递推关系
     * 3 确定DP数组的计算顺序
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }

        // 子问题：
        // f(k) = 偷 [0..k) 房间中的最大金额

        // f(0) = 0
        // f(1) = nums[0]
        // f(k) = max{ rob(k-1), nums[k-1] + rob(k-2) }
        int n = nums.length;
        int[] dp = new int[n+1];

        dp[0] = 0;
        dp[1] = nums[0];
        for (int k = 2; k <= n; k++) {
            dp[k] = Math.max(dp[k-1], nums[k-1] + dp[k-2]);
        }

        return dp[n];

    }

    /**
     * 最长回文字符串
     * 中心扩散法
     * @param s
     * @return
     */
    public String longestPalindrome1(String s) {

        if (s == null || s.length() == 0) {
            return "";
        }
        int strLen = s.length();
        int left = 0;
        int right = 0;
        int len = 1;
        int maxStart = 0;
        int maxLen = 0;

        for (int i = 0; i < strLen; i++) {
            left = i - 1;
            right = i + 1;
            while (left >= 0 && s.charAt(left) == s.charAt(i)) {
                len++;
                left--;
            }
            while (right < strLen && s.charAt(right) == s.charAt(i)) {
                len++;
                right++;
            }
            while (left >= 0 && right < strLen && s.charAt(right) == s.charAt(left)) {
                len = len + 2;
                left--;
                right++;
            }
            if (len > maxLen) {
                maxLen = len;
                maxStart = left;
            }
            len = 1;
        }
        return s.substring(maxStart + 1, maxStart + maxLen + 1);

    }

    /**
     * 只出现一次的数字
     * 使用位运算
     */
    public int singleNumber(int[] nums) {
        int x = 0;
        for (int num : nums)  // 1. 遍历 nums 执行异或运算
            x ^= num;
        return x;            // 2. 返回出现一次的数字 x
    }


    /**
     * 求众数，摩尔投票法
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        int x = 0, votes = 0;
        for (int num : nums){
            if (votes == 0) x = num;
            votes += num == x ? 1 : -1;
        }
        return x;
    }

    /**
     * 数组中的第K个最大元素
     */
    public int findKthLargest(int[] nums, int k) {
        int heapSize = nums.length;
        buildMaxHeap(nums, heapSize);
        for (int i = nums.length - 1; i >= nums.length - k + 1; --i) {
            swap(nums, 0, i);
            --heapSize;
            maxHeapify(nums, 0, heapSize);
        }
        return nums[0];
    }

    public void buildMaxHeap(int[] a, int heapSize) {
        for (int i = heapSize / 2; i >= 0; --i) {
            maxHeapify(a, i, heapSize);
        }
    }

    public void maxHeapify(int[] a, int i, int heapSize) {
        int l = i * 2 + 1, r = i * 2 + 2, largest = i;
        if (l < heapSize && a[l] > a[largest]) {
            largest = l;
        }
        if (r < heapSize && a[r] > a[largest]) {
            largest = r;
        }
        if (largest != i) {
            swap(a, i, largest);
            maxHeapify(a, largest, heapSize);
        }
    }














    public static void main(String[] args) {
        Solution s = new Solution();
        int[][] arr = new int[][]{
                {1, 4, 7, 11, 15},
                {2, 5, 8, 12, 19},
                {3, 6, 9, 16, 22},
                {10, 13, 14, 17, 24},
                {18, 21, 23, 26, 30}
        };

        boolean res = s.searchMatrix(arr,5);
        System.out.println(res);
    }
}

class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}

/**
 * LRU 缓存实现
 */
class LRUCache {
    class DLinkedNode {
        int key;
        int value;
        DLinkedNode prev;
        DLinkedNode next;
        public DLinkedNode() {}
        public DLinkedNode(int _key, int _value) {key = _key; value = _value;}
    }

    private Map<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
    private int size;
    private int capacity;
    private DLinkedNode head, tail;

    public LRUCache(int capacity) {
        this.size = 0;
        this.capacity = capacity;
        // 使用伪头部和伪尾部节点
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        DLinkedNode node = cache.get(key);
        if (node == null) {
            return -1;
        }
        // 如果 key 存在，先通过哈希表定位，再移到头部
        moveToHead(node);
        return node.value;
    }

    public void put(int key, int value) {
        DLinkedNode node = cache.get(key);
        if (node == null) {
            // 如果 key 不存在，创建一个新的节点
            DLinkedNode newNode = new DLinkedNode(key, value);
            // 添加进哈希表
            cache.put(key, newNode);
            // 添加至双向链表的头部
            addToHead(newNode);
            ++size;
            if (size > capacity) {
                // 如果超出容量，删除双向链表的尾部节点
                DLinkedNode tail = removeTail();
                // 删除哈希表中对应的项
                cache.remove(tail.key);
                --size;
            }
        }
        else {
            // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node.value = value;
            moveToHead(node);
        }
    }

    private void addToHead(DLinkedNode node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }

    private void removeNode(DLinkedNode node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void moveToHead(DLinkedNode node) {
        removeNode(node);
        addToHead(node);
    }

    private DLinkedNode removeTail() {
        DLinkedNode res = tail.prev;
        removeNode(res);
        return res;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode() {}
    TreeNode(int val) { this.val = val; }
    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

