<template>
  <div :class="['dropdown', { 'dropdown--active': show }]">
    <label :for="name">{{ label }}</label>
    <input
      class="dropdown__input"
      v-model="search"
      :id="name"
      :name="name"
      :disabled="disabled"
      :placeholder="placeholder"
      :autocomplete="'off'"
      @focus="focus"
      @blur="select(false)"
      @input="changed=true"
    />
    <div class="dropdown__content" v-show="show">
      <div v-for="(item, i) in filterList" :key="i" @mousedown="select(item)">
        {{ item.label }}
      </div>
      <div v-if="!filterList.length">Нет данных</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'd-autocomplete',
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
    value: String,
  },
  data() {
    return {
      selected: {},
      show: false,
      search: '',
      changed: null
    };
  },
  created() {
    this.search = this.value;
    this.changed = false;
  },
  computed: {
    filterList() {
      return this.list
        ? this.list.filter(item => {
            const search = this.search;
            return search && this.changed ? item.label.toLowerCase().includes(search.toLowerCase()) : true;
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
        this.$emit('change', item);
      } else {
        this.search = this.selected.label || this.value;
        this.show = false;
        this.changed = false;
      }
    },
    focus({ target }) {
      target.select()
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
.dropdown {
  position: relative;
  display: block;
  margin-bottom: 10px;
  padding: 0;
  &--active {
    .dropdown__input {
      border-radius: 4px 4px 0 0;
    }
    .dropdown__content {
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
    font-size: 1rem;
    font-weight: 400;
    border: 1px solid;
    border-radius: 4px;
    border: 1px solid #65B9F4;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
    width: 100%;
    color: #fff;
    background: #242f3d;
    cursor: pointer;
    display: block;
    z-index: 103;
    &:focus {
      border: 1px solid #65B9F4;
    }
    &:disabled {
      border: 1px solid #242F3D;
      cursor: default;
      opacity: 0.35;
    }
  }
  .dropdown__content {
    position: absolute;
    background-color: #242f3d;
    width: 100%;
    max-height: 190px;
    border: 1px solid #65B9F4;
    box-shadow: 0px -8px 34px 0px rgba(0, 0, 0, 0.05);
    overflow: auto;
    border-radius: 0 0 4px 4px;
    z-index: 102;
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
        color: #65B9F4;
        background-color: #1E2734;
      }
    }
  }
  // .dropdown:hover .dropdowncontent {
  //   display: block;
  // }
}
</style>
