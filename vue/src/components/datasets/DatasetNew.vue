<template>
	<div class="datasets">
		<p class="datasets__type">{{ list[selectedType] }} <i class="ci-icon ci-file" v-show="selectedType === 1" /></p>
		<div class="datasets__filter">
			<div class="datasets__filter--search">
				<DInputText small icon="search" v-model.trim="search" placeholder="Найти в списке"/>
				<div class="datasets__filter--search-results">
					<div v-for="(item, idx) in searchResults" :key="idx">{{ item.name }}</div>
				</div>
			</div>
			<div class="datasets__filter--sort">
				<div class="datasets__filter--sort-opt" v-show="cardsDisplay">
					<div class="datasets__filter--sort-opt--selected" @click="showSort(!showSortOpt)">
						<span>{{ selectedSort.title }}</span> <i class="ci-icon ci-chevron_down" />
					</div>
					<div class="options"
					v-show="showSortOpt"
					>
						<div v-for="(item, idx) in getSortOptions"
						:key="idx"
						@click="selectSort(item.idx)"
						>{{ item.title }}</div>
					</div>
				</div>
				<template v-if="selectedType !== 2">
					<i @click="cardsDisplay = true" :class="['ci-icon', 'ci-tile', { selected: cardsDisplay }]" />
					<i @click="cardsDisplay = false" :class="['ci-icon', 'ci-list', { selected: !cardsDisplay }]" />
				</template>
			</div>
		</div>
		<scrollbar class="datasets__scroll">
			<div v-if="cardsDisplay" class="datasets__cards">
				<CardDataset v-for="(item, idx) in sortedList" :key="idx" :dataset="item"/>
			</div>
			<div v-else class="datasets__table--wrapper">
				<table class="datasets__table">
					<thead>
						<tr>
							<th v-for="(item, idx) in getHeaders" :key="idx" @click="selectTableSort(item.idx)">
								<span>{{ item.title }}</span>
								<i v-show="selectedHeader === item.idx" :class="['ci-icon', `ci-thin_long_02_${reverseSort ? 'down': 'up'}`]"/>
							</th>
						</tr>
					</thead>
					<tbody>
						<tr v-for="(item, idx) in sortedTable" :key="'table'+idx">
							<td><i class="ci-icon ci-image" /> <span>{{ item.name }}</span></td>
							<td>{{ item.size ? item.size.short.toFixed(2) + ' ' + item.size.unit : 'Предустановленный' }}</td>
							<td>1 минуту назад</td>
							<td>{{ item.date ? item.date.toLocaleString() : '' }}</td>
						</tr>
					</tbody>
				</table>
			</div>
		</scrollbar>
	</div>
</template>

<script>
export default {
	components: {
		DInputText: () => import('@/components/global/design/forms/components/DInputText'),
		CardDataset: () => import('./cards/CardDatasetNew.vue')
	},
	data: () => ({
		list: ['Недавние датасеты', 'Проектные датасеты', 'Датасеты Terra'],
		search: '',
		cardsDisplay: true,
		showSortOpt: false,
		sortIdx: 0,
		sortOptions: [
			{
				title: 'По алфавиту от А до Я',
				value: 'alphabet',
				idx: 0
			},
			{
				title: 'По алфавиту от Я до А',
				value: 'alphabet_reverse',
				idx: 1
			},
			{
				title: 'Последние созданные',
				value: 'last_created',
				idx: 2
			},
			{
				title: 'Последние использованные',
				value: 'last_used',
				idx: 3
			},
			{
				title: 'Популярные',
				value: 'popular',
				idx: 4
			},
			{
				title: 'Последние добавленные',
				value: 'last_added',
				idx: 5
			},
		],
		theaders: [
			{
				title: 'Название',
				idx: 0
			},
			{
				title: 'Размер',
				idx: 1
			},
			{
				title: 'Автор',
				idx: 2
			},
			{
				title: 'Последнее использование',
				idx: 3
			},
			{
				title: 'Создание',
				idx: 4
			},
		],
		selectedHeader: 0,
		reverseSort: false
	}),
	props: ['datasets', 'selectedType'],
	computed: {
		datasetList() {
			return this.datasets
			.map(item => {
				if (item.date) {
					item.date = new Date(item.date)
				}
				return item
			})
			.filter(item => {
				if (this.selectedType === 1) return item.group === 'custom'
				if (this.selectedType === 2) return item.group !== 'custom'
			})
		},
		/* eslint-disable */
		sortedList() {
			if (this.selectedSort.value === 'alphabet') return this.datasetList.sort((a, b) => a.name.localeCompare(b.name))
			if (this.selectedSort.value === 'alphabet_reverse') return this.datasetList.sort((a, b) => b.name.localeCompare(a.name))
			return this.datasetList.sort((a, b) => b.date - a.date)
		},
		sortedTable() {
			const selectedSort = this.theaders.find(item => item.idx === this.selectedHeader)
			if (selectedSort.idx === 0) return this.reverseSort ?
				this.datasetList.sort((a, b) => b.name.localeCompare(a.name)):
				this.datasetList.sort((a, b) => a.name.localeCompare(b.name))
			if (selectedSort.idx === 1) return this.reverseSort ?
				this.datasetList.sort((a, b) => b.size.value - a.size.value):
				this.datasetList.sort((a, b) => a.size.value - b.size.value)
			if (selectedSort.idx === 2) return this.reverseSort ?
				this.datasetList.sort((a, b) => a.name.localeCompare(b.name)):
				this.datasetList.sort((a, b) => b.name.localeCompare(a.name))
			if (selectedSort.idx === 3) return this.reverseSort ?
				this.datasetList.sort((a, b) => a.date - b.date):
				this.datasetList.sort((a, b) => b.date - a.date)
			if (selectedSort.idx === 4) return this.reverseSort ?
				this.datasetList.sort((a, b) => a.date - b.date):
				this.datasetList.sort((a, b) => b.date - a.date)
		},
		/* eslint-enable */
		searchResults() {
			return this.search ?
			this.sortedList.filter(item => item.name.toLowerCase().includes(this.search.toLowerCase())): []
		},
		selectedSort() {
			return this.sortOptions.find(opt => opt.idx === this.sortIdx)
		},
		getSortOptions() {
			if (this.selectedType === 0) return this.sortOptions.slice(0, 4)
			if (this.selectedType === 1) return this.sortOptions.slice(0, 3)
			return this.sortOptions.slice(4, 6)
		},
		getHeaders() {
			const arr = [...this.theaders]
			if (this.selectedType === 1) {
				arr.splice(2, 1)
				return arr
			}
			return arr.slice(0, 4)
		}
	},
	methods: {
		showSort(val = false) {
			this.showSortOpt = val
		},
		selectSort(idx) {
			this.sortIdx = idx
			this.showSortOpt = false
		},
		selectTableSort(idx) {
			if (this.selectedHeader === idx) return this.reverseSort = !this.reverseSort
			this.selectedHeader = idx
		}
	},
	watch: {
		selectedType(idx) {
			if (idx === 2) {
				this.cardsDisplay = true
				return this.sortIdx = 4
			}
			this.sortIdx = 0
		}
	}
}
</script>

<style lang="scss" scoped>
.datasets {
	padding: 30px 0 0 40px;
	width: 100%;
	height: 100%;
	display: flex;
	flex-direction: column;
	gap: 30px;
	&__cards {
		display: flex;
		gap: 20px;
		flex-wrap: wrap;
		margin-bottom: 20px;
	}
	&__type {
		font-size: 14px;
		font-weight: 600;
		height: 20px;
	}
	&__filter {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-right: 60px;
		&--search {
			flex: 0 0 426px;
			position: relative;
			&-results {
				background: #233849;
				color: #A7BED3;
				box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.25);
				border-radius: 4px;
				position: absolute;
				top: calc(100% + 5px);
				width: 100%;
				z-index: 1;
				overflow: hidden auto;
				max-height: 440px;
				scrollbar-width: thin;
				scrollbar-color: #a7bed3 #233849;
				&::-webkit-scrollbar {
					width: 5px;
				}
				&::-webkit-scrollbar-thumb {
					background-color: #a7bed3;
					border-radius: 4px;
				}
				div {
					padding: 10px;
					cursor: pointer;
					overflow: hidden;
					text-overflow: ellipsis;
					white-space: nowrap;
					&:hover {
						background-color: #1E2734;
						color: #65B9F4;
					}
				}
			}
		}
		&--sort {
			display: flex;
			gap: 20px;
			align-items: center;
			&-opt {
				color: #A7BED3;
				font-size: 14px;
				position: relative;
				cursor: pointer;
				min-width: 220px;
				user-select: none;
				&--selected {
					display: flex;
					align-items: center;
					gap: 10px;
					justify-content: flex-end;
				}
				.options {
					position: absolute;
					top: calc(100% + 10px);
					background-color: #242F3D;
					z-index: 1;
					border-radius: 4px;
					overflow: hidden;
					width: 100%;
					div {
						padding: 10px;
						&:hover {
							color: #65B9F4;
							background-color: #1E2734;
						}
					}
				}
			}
			i {
				color: #A7BED3;
				font-size: 20px;
				cursor: pointer;
			}
		}
	}
	&__scroll {
		justify-self: stretch;
	}
	&__table {
		font-size: 14px;
		font-weight: 400;
		width: 100%;
		&--wrapper {
			width: calc(100% - 150px);
			position: relative;
		}
		tr > *:nth-child(2) {
			text-align: right;
		}
		thead {
			background-color: #17212B;
			color: #6C7883;
			position: sticky;
			top: 0;
			i {
				color: #65B9F4;
				font-size: 20px;
				vertical-align: middle;
			}
			tr {
				height: 35px;
			}
			th {
				font-weight: inherit;
				padding: 0 50px;
				min-width: 150px;
				user-select: none;
				&:first-child {
					padding: 15px 10px;
				}
				* {
					vertical-align: middle;
					cursor: pointer;
				}
			}
		}
		tbody {
			tr {
				height: 55px;
				cursor: pointer;
				&:hover {
					background-color: #0E1621;
				}
			}
			td {
				color: #F2F5FA;
				padding: 15px 50px;
				white-space: nowrap;
				text-overflow: ellipsis;
				overflow: hidden;
				max-width: 450px;
				&:first-child {
					padding: 15px 10px;
				}
				i {
					font-size: 19px;
					color: #6C7883;
					margin-right: 15px;
				}
				* {
					vertical-align: middle;
				}
			}
		}
	}
}

.ci-tile {
	display: inline-block;
	background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTE4LjI1IDE4LjI1SDEzLjc1VjEzLjc1SDE4LjI1VjE4LjI1Wk0xMC4yNSAxOC4yNUg1Ljc1VjEzLjc1SDEwLjI1VjE4LjI1Wk0xOC4yNSAxMC4yNUgxMy43NVY1Ljc1SDE4LjI1VjEwLjI1Wk0xMC4yNSAxMC4yNUg1Ljc1VjUuNzVIMTAuMjVWMTAuMjVaIiBzdHJva2U9IiNBN0JFRDMiIHN0cm9rZS13aWR0aD0iMS41Ii8+Cjwvc3ZnPgo=');
	width: 24px;
	height: 24px;
	border-radius: 4px;
	&.selected {
		background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTE4LjI1IDE4LjI1SDEzLjc1VjEzLjc1SDE4LjI1VjE4LjI1Wk0xMC4yNSAxOC4yNUg1Ljc1VjEzLjc1SDEwLjI1VjE4LjI1Wk0xOC4yNSAxMC4yNUgxMy43NVY1Ljc1SDE4LjI1VjEwLjI1Wk0xMC4yNSAxMC4yNUg1Ljc1VjUuNzVIMTAuMjVWMTAuMjVaIiBzdHJva2U9IiM2NUI5RjQiIHN0cm9rZS13aWR0aD0iMS41Ii8+CjxwYXRoIGQ9Ik00IDFIMjBWLTFINFYxWk0yMyA0VjIwSDI1VjRIMjNaTTIwIDIzSDRWMjVIMjBWMjNaTTEgMjBWNEgtMVYyMEgxWk00IDIzQzIuMzQzMTUgMjMgMSAyMS42NTY5IDEgMjBILTFDLTEgMjIuNzYxNCAxLjIzODU4IDI1IDQgMjVWMjNaTTIzIDIwQzIzIDIxLjY1NjkgMjEuNjU2OSAyMyAyMCAyM1YyNUMyMi43NjE0IDI1IDI1IDIyLjc2MTQgMjUgMjBIMjNaTTIwIDFDMjEuNjU2OSAxIDIzIDIuMzQzMTUgMjMgNEgyNUMyNSAxLjIzODU4IDIyLjc2MTQgLTEgMjAgLTFWMVpNNCAtMUMxLjIzODU4IC0xIC0xIDEuMjM4NTggLTEgNEgxQzEgMi4zNDMxNSAyLjM0MzE1IDEgNCAxVi0xWiIgZmlsbD0iIzY1QjlGNCIvPgo8L3N2Zz4K');
	}
}

.ci-list {
	display: inline-block;
	width: 24px;
	height: 24px;
	border-radius: 4px;
	background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIwIDE5SDRWMTdIMjBWMTlaTTIwIDE1SDRWMTNIMjBWMTVaTTIwIDExSDRWOUgyMFYxMVpNMjAgN0g0VjVIMjBWN1oiIGZpbGw9IiNBN0JFRDMiLz4KPC9zdmc+Cg==');
	&.selected {
		background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIwIDE5SDRWMTdIMjBWMTlaTTIwIDE1SDRWMTNIMjBWMTVaTTIwIDExSDRWOUgyMFYxMVpNMjAgN0g0VjVIMjBWN1oiIGZpbGw9IiM2NUI5RjQiLz4KPHBhdGggZD0iTTQgMUgyMFYtMUg0VjFaTTIzIDRWMjBIMjVWNEgyM1pNMjAgMjNINFYyNUgyMFYyM1pNMSAyMFY0SC0xVjIwSDFaTTQgMjNDMi4zNDMxNSAyMyAxIDIxLjY1NjkgMSAyMEgtMUMtMSAyMi43NjE0IDEuMjM4NTggMjUgNCAyNVYyM1pNMjMgMjBDMjMgMjEuNjU2OSAyMS42NTY5IDIzIDIwIDIzVjI1QzIyLjc2MTQgMjUgMjUgMjIuNzYxNCAyNSAyMEgyM1pNMjAgMUMyMS42NTY5IDEgMjMgMi4zNDMxNSAyMyA0SDI1QzI1IDEuMjM4NTggMjIuNzYxNCAtMSAyMCAtMVYxWk00IC0xQzEuMjM4NTggLTEgLTEgMS4yMzg1OCAtMSA0SDFDMSAyLjM0MzE1IDIuMzQzMTUgMSA0IDFWLTFaIiBmaWxsPSIjNjVCOUY0Ii8+Cjwvc3ZnPgo=');
	}
}

.ci-file {
	display: inline-block;
	width: 20px;
	height: 20px;
	margin-left: 10px;
	vertical-align: middle;
	background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzI2NDQ6NDQzMDYpIj4KPHBhdGggZD0iTTE1LjAwMDIgMTguMzMzM0g1LjAwMDE2QzQuMDc5NjkgMTguMzMzMyAzLjMzMzUgMTcuNTg3MSAzLjMzMzUgMTYuNjY2NlYzLjMzMzI5QzMuMzMzNSAyLjQxMjgyIDQuMDc5NjkgMS42NjY2MyA1LjAwMDE2IDEuNjY2NjNIMTAuODMzNUMxMC44NDI2IDEuNjY3MDIgMTAuODUxNiAxLjY2ODcgMTAuODYwMiAxLjY3MTYzQzEwLjg2ODMgMS42NzQxNCAxMC44NzY3IDEuNjc1ODIgMTAuODg1MiAxLjY3NjYzQzEwLjk1ODcgMS42ODEzNCAxMS4wMzEyIDEuNjk1NjIgMTEuMTAxIDEuNzE5MTNMMTEuMTI0MyAxLjcyNjYzQzExLjE0MzIgMS43MzMwMyAxMS4xNjE2IDEuNzQwODMgMTEuMTc5MyAxLjc0OTk2QzExLjI3MDEgMS43OTAzMSAxMS4zNTI4IDEuODQ2NzggMTEuNDIzNSAxLjkxNjYzTDE2LjQyMzUgNi45MTY2M0MxNi40OTMzIDYuOTg3MjggMTYuNTQ5OCA3LjA3MDAxIDE2LjU5MDIgNy4xNjA3OUMxNi41OTg1IDcuMTc5MTMgMTYuNjA0MyA3LjE5ODI5IDE2LjYxMSA3LjIxNzQ2TDE2LjYxODUgNy4yMzkxM0MxNi42NDE4IDcuMzA4NjIgMTYuNjU1NSA3LjM4MDk0IDE2LjY1OTMgNy40NTQxM0MxNi42NjA2IDcuNDYxNzggMTYuNjYyNiA3LjQ2OTMxIDE2LjY2NTIgNy40NzY2M0MxNi42NjY2IDcuNDg0MzEgMTYuNjY3MiA3LjQ5MjE0IDE2LjY2NjkgNy40OTk5NlYxNi42NjY2QzE2LjY2NjkgMTcuNTg3MSAxNS45MjA2IDE4LjMzMzMgMTUuMDAwMiAxOC4zMzMzWk01LjAwMDE2IDMuMzMzMjlWMTYuNjY2NkgxNS4wMDAyVjguMzMzMjlIMTAuODMzNUMxMC4zNzMzIDguMzMzMjkgMTAuMDAwMiA3Ljk2MDIgMTAuMDAwMiA3LjQ5OTk2VjMuMzMzMjlINS4wMDAxNlpNMTEuNjY2OCA0LjUxMTYzVjYuNjY2NjNIMTMuODIxOEwxMS42NjY4IDQuNTExNjNaIiBmaWxsPSIjQTdCRUQzIi8+CjxwYXRoIGQ9Ik02LjE2NjUgOC42NjY2M0gtMC44MzM0OTZILTEuMzMzNVY5LjE2NjYzVjEwLjgzMzNWMTEuMzMzM0gtMC44MzM0OTZINi4xNjY1VjEzLjMzMzNWMTQuMzczNkw2Ljk3ODg1IDEzLjcyMzdMMTEuMTQ1NSAxMC4zOTA0TDExLjYzMzYgOS45OTk5NkwxMS4xNDU1IDkuNjA5NTJMNi45Nzg4NSA2LjI3NjE5TDYuMTY2NSA1LjYyNjMxVjYuNjY2NjNWOC42NjY2M1oiIGZpbGw9IiNBN0JFRDMiIHN0cm9rZT0iIzE3MjEyQiIvPgo8L2c+CjxkZWZzPgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzI2NDQ6NDQzMDYiPgo8cmVjdCB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIGZpbGw9IndoaXRlIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg==');
}
</style>