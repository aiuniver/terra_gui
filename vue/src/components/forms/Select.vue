<template>
  <div class="field">
    <label class="field__label">{{ label }}</label>
    <input style="display: none" :name="parse" :value="select"/>
    <at-select
      v-model="select"
      clearable
      size="small"
      style="width: 100px"
      @on-change="change"
    >
      <at-option v-for="({ label, value }, key) in items" :key="'item_' + key" :value="value">
        {{ label }}
      </at-option>
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
        return this.lists.map((i) => {
          return i || ''
        })
      } else {
        return Object.keys(this.lists);
      }
    },
  },
  methods: {
    change(value) {
      this.$emit('input', value)
      this.$emit('change', { name: this.name, value })
      
    // bus.$emit("change", e);
    },
  },
  created() {
    this.select = this.value

    // console.log('created', this.select)
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
.field {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  align-items: center;
  &__label {
    width: auto;
    padding: 0 20px 0 10px;
    text-align: left;
    color: #A7BED3;
    display: block;
    margin: 0;
    line-height: 1.25;
    font-size: .75rem;
  }
  &__input {
    height: 22px;
    font-size: 0.75rem;
    max-width: 100px;
    width: 100px;
  }
}
</style>
