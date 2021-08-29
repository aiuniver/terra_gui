<template>
  <div class="params__navbar">
    <ul class="params__navbar--items">
      <template v-for="({ title, active }, i) of items">
        <li
          :key="'item_' + i"
          :class="['params__navbar--item', { active: active }]"
          :title="title"
          @click="click(i)"
        >
          {{ title }}
        </li>
      </template>
    </ul>
  </div>
</template>


<script>
export default {
  name: "PNavbar",
  props: {
    list: {
      type: Array,
      default: () => [
        { title: "Слой", active: true },
      ],
    },
  },
  data: () => ({
    items: [],
  }),
  methods: {
    click(id) {
      this.items = this.items.map((item, i) => {
        item.active = (id === i)
        return item;
      })
    }
  },
  created() {
    this.items = this.list;
  },
};
</script>

<style lang="scss" scoped>
.params__navbar {
  &--items {
    background-color: #0e1621;
    display: -webkit-box;
    display: -moz-box;
    display: -ms-flexbox;
    display: -webkit-flex;
    display: flex;
    -webkit-box-direction: normal;
    -moz-box-direction: normal;
    -webkit-box-orient: horizontal;
    -moz-box-orient: horizontal;
    -webkit-flex-direction: row;
    -ms-flex-direction: row;
    flex-direction: row;
    -webkit-flex-wrap: nowrap;
    -ms-flex-wrap: nowrap;
    flex-wrap: nowrap;
    -webkit-box-pack: start;
    -moz-box-pack: start;
    -webkit-justify-content: flex-start;
    -ms-flex-pack: start;
    justify-content: flex-start;
    -webkit-align-content: flex-start;
    -ms-flex-line-pack: start;
    align-content: flex-start;
    -webkit-box-align: center;
    -moz-box-align: center;
    -webkit-align-items: center;
    -ms-flex-align: center;
    align-items: center;
  }
  &--item {
    display: block;
    line-height: 20px;
    padding: 0 20px;
    font-size: 0.75rem;
    white-space: nowrap;
    cursor: pointer;
    transition: color 0.3s ease-in-out;
    user-select: none;
  }
}
</style>