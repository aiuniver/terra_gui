<template>
  <div
    :class="['dataset-card-item', { active: dataset.active, selected: loaded }]"
    @click="$emit('clickCard', dataset, cardIndex)"
  >
    <div class="dataset-card">
      <div class="card-title">{{ dataset.name }}</div>
      <div class="card-paggination">
        <div @click="next">next</div>
        <div @click="prev">prev</div>
      </div>
      <div class="card-body">
        <div v-for="({ name }, key) of activeCard" :key="`tag_${key}`" class="card-tag">
          {{ name }}
        </div>
      </div>

      <div :class="'card-extra ' + (dataset.size ? 'is-custom' : '')">
        <div class="wrapper">
          <span>{{ dataset.size ? dataset.size : 'Предустановленный' }}</span>
        </div>
        <div class="remove"></div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    dataset: {
      type: Object,
      default: () => {},
    },
    cardIndex: {
      type: Number,
    },
    loaded: {
      type: Boolean,
      default: false,
    },
  },

  data: () => ({
    paggination: {
      totalPages: 0,
      currentPage: 1,
      to: 0,
      from: 0,
      countView: 4,
    },

    activeCard: [],
  }),

  created() {
    this.paggination.totalPages = Math.ceil(this.dataset.tags.length / this.paggination.countView);
    this.paggination.to =
      this.dataset.tags.length > this.paggination.countView ? this.paggination.countView : this.dataset.tags.length;
    this.paginateCard(0, this.paggination.to);
  },

  methods: {
    paginateCard(from, to) {
      this.activeCard = this.dataset.tags.slice(from, to);
    },
    next() {
      if (this.paggination.totalPages <= this.paggination.currentPage) return;
      this.paggination.currentPage++;
      this.paggination.from += this.paggination.countView;
      if (this.paggination.to + this.paggination.countView > this.dataset.tags.length)
        this.paggination.to += this.dataset.tags.length;
      else this.paggination.to += this.paggination.countView;
      this.paginateCard(this.paggination.from, this.paggination.to);
    },
    prev() {
      if (1 >= this.paggination.currentPage) return;
      this.paggination.currentPage--;
      this.paggination.from -= this.paggination.countView;
      this.paggination.to -= this.paggination.countView + 1;
      this.paginateCard(this.paggination.from, this.paggination.to);
    },
  },
};
</script>

<style lang="scss">
// For tests
.card-paggination {
  display: flex;
  margin-top: 30px;
  div {
    margin-right: 5px;
  }
}

.card-body {
  justify-content: inherit;
}

// For tests
</style>
