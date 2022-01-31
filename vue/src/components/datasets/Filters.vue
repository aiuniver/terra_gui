<template>
  <div class="filters" ref="filters">
    <div class="title my-4">Теги</div>
    <div class="inner">
      <ul>
        <li v-for="({ name, alias, active }, i) in tags" :key="i" @click="click(i, alias)" :class="{ active }">
          <span>
            {{ name }}
          </span>
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
export default {
  computed: {
    tags: {
      set(value) {
        this.$store.dispatch('datasets/setTags', value);
      },
      get() {
        return this.$store.getters['datasets/getTags'];
      },
    },
    tagsFilter: {
      set(value) {
        this.$store.dispatch('datasets/setTagsFilter', value);
      },
      get() {
        return this.$store.getters['datasets/getTagsFilter'];
      },
    },
  },
  methods: {
    click(i) {
      this.tags[i].active = !this.tags[i].active;
      this.tagsFilter = this.tags.reduce((t, { active, alias }) => {
        if (active) {
          t.push(alias);
        }
        return t;
      }, []);
    },
    myEventHandler() {
      this.$store.dispatch('settings/setHeight', { filter: this.$refs.filters.clientHeight });
    },
  },
  watch: {
    tags() {
      this.$nextTick(() => {
        this.$store.dispatch('settings/setHeight', { filter: this.$refs.filters.clientHeight });
      });
    },
  },
  mounted() {
    setTimeout(() => {
      this.$store.dispatch('settings/setHeight', { filter: this.$refs.filters.clientHeight });
    }, 100);
  },
  created() {
    window.addEventListener('resize', this.myEventHandler);
  },
  destroyed() {
    window.removeEventListener('resize', this.myEventHandler);
  },
};
</script>

<style lang="scss">
.filters {
  padding: 0 20px;
  ul {
    margin: 0;
    list-style: none;
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: flex-start;
    align-content: flex-start;
    align-items: flex-start;
  }
  li {
    padding: 0 5px 10px 5px;
  }
  span {
    display: block;
    line-height: 18px;
    padding: 0 15px;
    font-size: 0.75rem;
    border: 1px solid;
    border-color: #6c7883;
    border-radius: 4px;
    white-space: nowrap;
    cursor: pointer;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
    transition: border-color 0.3s ease-in-out;
  }
  .active span {
    border-color: #65b9f4;
  }
}
</style>