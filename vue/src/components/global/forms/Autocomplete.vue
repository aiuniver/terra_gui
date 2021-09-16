<template>
  <div :class="['auto-complete', { 'auto-complete--active': show }]">
    <label :for="name">{{ label }}</label>
    <input
      class="auto-complete__input"
      :class="['auto-complete__input', { 'auto-complete__error': error }]"
      v-model="search"
      :id="name"
      :name="name"
      :disabled="disabled"
      :placeholder="placeholder"
      :autocomplete="'off'"
      @focus="focus"
      @blur="select(false)"
      @mouseover="hover = true"
      @mouseleave="hover = false"
    />
    <div class="auto-complete__content" v-show="show">
      <div v-for="(item, i) in filterList" :key="i" @mousedown="select(item)">
        {{ item.label }}
      </div>
      <div v-if="!filterList.length">Нет данных</div>
    </div>
    <div v-if="error && hover" class="auto-complete__hint">
      <span>{{ error }}</span>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-auto-complete',
  props: {
    name: String,
    list: {
      type: Array,
      required: true,
      default: () => [],
    },
    placeholder: String,
    disabled: Boolean,
    label: String,
    parse: String,
    value: String,
    error: String,
  },
  data() {
    return {
      selected: {},
      show: false,
      search: '',
      hover: false,
    };
  },
  created() {
    // this.$emit("selected", { name: this.value });
    // this.$emit('parse', { parse: this.parse, value: this.value });
    this.search = this.value;
  },
  computed: {
    filterList() {
      return this.list
        ? this.list.filter(item => {
            const search = this.search;
            return search ? item.label.toLowerCase().includes(search.toLowerCase()) : true;
          })
        : [];
    },
  },
  methods: {
    select(item) {
      // console.log(item);
      if (item) {
        this.selected = item;
        this.show = false;
        this.search = item.label;
        this.$emit('input', this.selected.value);
        this.$emit('change', { name: this.name, value: item.value });
        this.$emit('parse', { name: this.name, parse: this.parse, value: item.value });
      } else {
        this.search = this.selected.label || this.value;
        this.show = false;
      }
    },
    focus({ target }) {
      target.select();
      this.show = true;
      this.$emit('focus', true);
    },
  },
  watch: {
    value: {
      handler(value) {
        // console.log(value)
        this.show = false;
        this.search = value;
      },
    },
  },
};
</script>

<style lang="scss" scoped>
.auto-complete {
  position: relative;
  display: block;
  margin-bottom: 10px;
  padding: 0;
  &--active {
    .auto-complete__input {
      border-radius: 4px 4px 0 0;
    }
    .auto-complete__content {
      border-top: 0 !important;
    }
  }
  label {
    color: #a7bed3;
    display: block;
    margin: 0 0 10px 0;
    line-height: 1;
    font-size: 0.75rem;
  }
  &__input {
    height: 42px;
    padding: 5px 10px;
    font-size: 0.875rem;
    font-weight: 400;
    border: 1px solid;
    border-radius: 4px;
    border-color: #6c7883;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
    width: 100%;
    color: #fff;
    background: #242f3d;
    cursor: pointer;
    display: block;
    z-index: 101;
    &:focus {
      border: 1px solid #e7ecf5;
    }
    &:disabled {
      border: 1px solid #6c7883;
      cursor: auto;
      opacity: 0.35;
    }
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
  &__error {
    border-color: #ca5035;
    color: #ca5035;
  }
  .auto-complete__content {
    bottom: -28px;
    position: absolute;
    background-color: #242f3d;
    width: 100%;
    max-height: 190px;
    border: 1px solid #6c7883;
    box-shadow: 0px -8px 34px 0px rgba(0, 0, 0, 0.05);
    overflow: auto;
    border-radius: 0 0 4px 4px;
    z-index: 103;
    // bottom: -29px;
    > div {
      color: #a7bed3;
      font-size: 0.7em;
      line-height: 1em;
      padding: 8px;
      text-decoration: none;
      display: block;
      cursor: pointer;
      &:hover {
        color: #e7ecf5;
        background-color: #6c7883;
      }
    }
  }
  // .auto-complete:hover .dropdowncontent {
  //   display: block;
  // }
}
</style>
