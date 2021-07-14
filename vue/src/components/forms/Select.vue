<template>
  <div class="field-form field-inline field-reverse">
    <label>{{ label }}</label>
    <input style="display: none" :name="parse" :value="value"/>
    <at-select
      clearable
      size="small"
      :value="value"
      style="width: 100px"
      @on-change="change"
    >
      <at-option v-for="(item, key) in items" :key="item+key" :value="item">{{
        item
      }}</at-option>
    </at-select>
  </div>
</template>

<script>
// import { bus } from '@/main'
export default {
  name: "Select",
  props: {
    label: {
      type: String,
      default: "Label",
    },
    type: {
      type: String,
      default: "",
    },
    value: {
      type: [String, Number],
    },
    name: {
      type: String,
    },
    parse: {
      type: String,
    },
    lists: {
      type: [Array, Object],
    },
  },
  data: () => ({
    select: "",
  }),
  computed: {
    items () {
      if (Array.isArray(this.lists)) {
        return this.lists
      } else {
        return Object.keys(this.lists);
      }
    },
  },
  methods: {
    change(e) {
      console.log(e)
      this.$emit('input', e)
    // bus.$emit("change", e);
    },
  },
  created() {
    this.select = this.value
    console.log(this.select)
    // console.log('created', this.name);
    // bus.$on("change", () => {
    //   console.log(this.name, 'data');
    // });
  },
  destroyed() {
    // bus.$off()
    // console.log('destroyed', this.name);
  },
};
</script>


<style lang="scss" scope>
</style>
