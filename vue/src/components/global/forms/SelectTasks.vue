<template>
  <div class="t-field">
    <label class="t-field__label" :style="'max-width: ' + maxLabel + 'px'">{{ label }}</label>
    <input style="display: none" :name="parse" :value="select" />
    <at-select
      :value="select"
      :class="['t-field__select', { 't-field__error': error }]"
      size="small"
      style="width: 100px"
      @on-change="change"
      @click="cleanError"
      :disabled="disabled"
      @mouseover="hover = true"
      @mouseleave="hover = false"
    >
      <at-option v-for="({ label, value }, key) in filter" :key="'item_' + key" :value="value" :title="label">
        {{ label }}
      </at-option>
    </at-select>
    <div v-if="error && hover" class="t-field__hint">
      <span>{{ error }}</span>
    </div>
  </div>
</template>

<script>
// import { bus } from '@/main'
export default {
  name: 't-select-tasks',
  props: {
    label: {
      type: String,
      default: 'Label',
    },
    type: {
      type: String,
      default: '',
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

    disabled: Boolean,
    error: String,
    maxLabel: {
      type: Number,
      default: 130,
    },
  },
  data: () => ({
    hover: false,
    select: '',
    isChange: false
  }),
  computed: {
    filter() {
      if (this.dropFiles.length) {
        if (this.dropFiles.find(item => item.type === 'table')) {
          return this.items.filter(item => item.value === 'Dataframe');
        } else {
          return this.items.filter(item => item.value !== 'Dataframe');
        }
      } else {
        return this.items;
      }
    },
    dropFiles: {
      set(value) {
        this.$store.dispatch('datasets/setFilesDrop', value);
      },
      get() {
        return this.$store.getters['datasets/getFilesDrop'];
      },
    },
    items() {
      return this.lists
        ? Array.isArray(this.lists)
          ? this.lists.map(item => item || '')
          : Object.keys(this.lists)
        : [];
    },
  },
  methods: {
    cleanError() {
      if (this.error) {
        this.$emit('cleanError');
      }
    },
    change(value) {
      this.$emit('input', value);
      this.$emit('change', { name: this.name, value });
      this.$emit('parse', { name: this.name, parse: this.parse, value });
      // bus.$emit("change", e);
    },
  },
  created() {
    this.select = this.value;

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
  watch: {
    filter(value) {
      const type = value.find(item => item);
      if (type) {
        this.select = type.value;
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-field {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  align-items: center;
  position: relative;
  &__label {
    width: 150px;
    max-width: 130px;
    padding: 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    // white-space: nowrap;
    overflow: hidden;
  }
  &__input {
    height: 24px;
    font-size: 0.75rem;
    max-width: 109px;
    width: 109px;
  }
  &__select {
    flex: 0 0 100px;
  }
  &__error {
    border-color: #b53b3b;
    color: #ca5035;
  }
  &__hint {
    user-select: none;
    position: absolute;
    height: 22px;
    display: flex;
    align-items: center;
    padding: 0 5px 0 5px;
    top: 25px;
    background-color: #ca5035;
    color: #fff;
    border-radius: 4px;
    z-index: 5;
    // display: none;
    span {
      font-style: normal;
      font-weight: normal;
      font-size: 9px;
      line-height: 12px;
    }
    // &--hover {
    //   display: flex;
    // }
  }
}
</style>
