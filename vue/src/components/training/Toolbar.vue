<template>
  <div class="toolbar">
    <ul class="toolbar__menu">
      <li
        v-for="({ title, disabled, icon }, index) of items"
        :key="'items_' + index"
        :class="['toolbar__menu--item']"
        :disabled="disabled"
        @click="click(index)"
      >
        <i :title="title" :class="['t-icon', icon, { active: collapse.includes(index.toString()) }]" />
      </li>
    </ul>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
export default {
  name: 'Toolbar',
  data: () => ({}),
  computed: {
    ...mapGetters({
      items: 'trainings/getToolbar',
    }),
    collapse: {
      set(value) {
        this.$store.dispatch('trainings/setСollapse', value);
      },
      get() {
        return this.$store.getters['trainings/getСollapse'];
      },
    },
  },
  created() {
    console.log(this.$store.getters['trainings/getСollapse']);
  },
  methods: {
    click(value) {
      const index = value.toString();
      if (this.collapse.includes(index)) {
        this.collapse = this.collapse.filter(item => item !== index);
      } else {
        this.collapse.push(index);
      }
    },
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/variables/default.scss";
.toolbar {
  z-index: 10;
  width: 41px;
  flex-shrink: 0;
  position: relative;
  border-right: #0e1621 1px solid;
  &__menu {
    padding: 10px 0;
    list-style: none;
    &--item {
      width: 40px;
      height: 40px;
      width: 40px;
      height: 40px;
      display: flex;
      justify-content: center;
      align-items: center;
      &[disabled='disabled'] {
        opacity: 0.1;
        cursor: default;
      }
    }
  }
}
</style>