<template>
  <div class="dropdown" v-if="options">
    <!-- Dropdown Input -->
    <label>{{ label }}</label>
    <input
      class="dropdown-input"
      :name="name"
      @focus="showOptions"
      @blur="exit()"
      @keyup="keyMonitor"
      v-model="searchFilter"
      :disabled="disabled"
      :placeholder="placeholder"
    />

    <!-- Dropdown Menu -->
    <div class="dropdown-content" v-show="optionsShown">
      <div
        class="dropdown-item"
        @mousedown="selectOption(option)"
        v-for="(option, index) in filteredOptions"
        :key="index"
      >
        {{ option.name || option.id || "-" }}
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "Dropdown",
  template: "Dropdown",
  props: {
    name: {
      type: String,
      required: false,
      default: "dropdown",
    },
    options: {
      type: Array,
      required: true,
      default: () => [],
    },
    placeholder: {
      type: String,
      required: false,
      default: "Please select an option",
    },
    disabled: {
      type: Boolean,
      required: false,
      default: false,
    },
    maxItem: {
      type: Number,
      required: false,
      default: 6,
    },
    label: {
      type: String,
      default: ''
    }
  },
  data() {
    return {
      selected: {},
      optionsShown: false,
      searchFilter: "",
    };
  },
  created() {
    this.$emit("selected", this.selected);
  },
  computed: {
    filteredOptions() {
      const filtered = [];
      const regOption = new RegExp(this.searchFilter, "ig");
      for (const option of this.options) {
        if (this.searchFilter.length < 1 || option.name.match(regOption)) {
          if (filtered.length < this.maxItem) filtered.push(option);
        }
      }
      return filtered;
    },
  },
  methods: {
    selectOption(option) {
      this.selected = option;
      this.optionsShown = false;
      this.searchFilter = this.selected.name;
      this.$emit("selected", this.selected);
    },
    showOptions(e) {
      if (!this.disabled) {
        this.searchFilter = "";
        this.optionsShown = true;
        this.$emit("focus", e);
      }
    },
    exit() {
      if (!this.selected.id) {
        this.selected = {};
        this.searchFilter = "";
      } else {
        this.searchFilter = this.selected.name;
      }
      this.$emit("selected", this.selected);
      this.optionsShown = false;
    },
    // Selecting when pressing Enter
    keyMonitor: function (event) {
      if (event.key === "Enter" && this.filteredOptions[0])
        this.selectOption(this.filteredOptions[0]);
    },
  },
  watch: {
    searchFilter() {
      if (this.filteredOptions.length === 0) {
        this.selected = {};
      } else {
        this.selected = this.filteredOptions[0];
      }
      this.$emit("filter", this.searchFilter);
    },
  },
};
</script>


<style lang="scss" scope>
.dropdown {
  position: relative;
  display: block;
  margin: auto;
  label {
    color: #a7bed3;
    display: block;
    margin: 0 0 10px 0;
    line-height: 1.25;
    font-size: .75rem;
  }
  .dropdown-input {
    height: 42px;
    padding: 0 10px;
    font-size: .875rem;
    font-weight: 400;
    border: 1px solid;
    border-radius: 4px;
    border-color: #6C7883;
    transition: border-color .3s ease-in-out, opacity .3s ease-in-out;
    width: 100%;
    color: #fff;
    border-color: #6c7883;
    background: #242f3d;
    cursor: pointer;
    // border: 1px solid #e7ecf5;
    border-radius: 3px;
    display: block;
    font-size: 0.8em;
    padding: 6px;
    &:focus {
      // background: #f8f8fa;
      border: 1px solid #e7ecf5;
    }
  }
  .dropdown-content {
    position: absolute;
    background-color: #242f3d;
    width: 90%;
    max-height: 248px;
    border: 1px solid #6c7883;
    box-shadow: 0px -8px 34px 0px rgba(0, 0, 0, 0.05);
    overflow: auto;
    z-index: 102;
    .dropdown-item {
      color: #A7BED3;
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
  .dropdown:hover .dropdowncontent {
    display: block;
  }
}
</style>