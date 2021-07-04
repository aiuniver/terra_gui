<template>
  <div class="field-form field-inline field-reverse">
    <label>{{ label }}</label>
    <input
      style="display: none"
      :name="`${parse}[parameters][${name}]`"
      :value="select"
    />
    <at-select v-model="select" clearable size="small" :value="value" style="width: 100px">
      <at-option v-for="(item, i) of lists" :key="i" :value="item">{{ item }}</at-option>
    </at-select>
  </div>
</template>

<script>
// import { bus } from '@/main'
export default {
  name: "Dropdown",
  props: {
    label: {
      type: String,
      default: "Label",
    },
    type: {
      type: String,
      default: "text",
    },
    value: {
      type: [String, Number],
    },
    name: {
      type: String,
    },
    parse: {
      type: String
    },
    lists: {
      type: Array,
      default: () => []
    }
  },
  data: () => ({
    select: ''
  }),
  methods: {
    // change(e) {
      // bus.$emit("change", e);
    // },
  },
  created() {
    // console.log('created', this.name);
    // bus.$on("change", () => {
    //   console.log(this.name, 'data');
    // });
  },
  destroyed() {
    // bus.$off()
    // console.log('destroyed', this.name);
  }
}
</script>


<style lang="scss" scope>

</style>
