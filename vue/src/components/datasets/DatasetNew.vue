<template>
	<div class="datasets">
		<p class="datasets__type">{{ list[selectedType] }} <i class="ci-icon ci-file" v-show="selectedType === 1" /></p>
		<div class="datasets__filter">
			<DInputText small icon="search" v-model.trim="search" class="datasets__filter--search" placeholder="Найти в списке"/>
			<div class="datasets__filter--sort">
				<p v-show="cardsDisplay"><span>Последние просмотренные</span> <i class="ci-icon ci-chevron_down" /></p>
				<i @click="cardsDisplay = true" :class="['ci-icon', 'ci-tile', { selected: cardsDisplay }]" />
				<i @click="cardsDisplay = false" :class="['ci-icon', 'ci-list', { selected: !cardsDisplay }]" />
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
							<th>Название</th>
							<th>Размер</th>
							<th>Последнее использование</th>
							<th>Создание</th>
						</tr>
					</thead>
					<tbody>
						<tr v-for="(item, idx) in sortedList" :key="'table'+idx">
							<td><i class="ci-icon ci-image" /> <span>{{ item.name }}</span></td>
							<td>{{ item.size ? item.size.short.toFixed(2) + ' ' + item.size.unit : 'Предустановленный' }}</td>
							<td>1 минуту назад</td>
							<td>1 минуту назад</td>
						</tr>
					</tbody>
				</table>
			</div>
		</scrollbar>
	</div>
</template>

<script>
import CardDataset from './cards/CardDatasetNew.vue'

export default {
	components: {
		CardDataset,
		DInputText: () => import('@/components/global/design/forms/components/DInputText')
	},
	data: () => ({
		list: ['Недавние датасеты', 'Проектные датасеты', 'Датасеты Terra'],
		search: '',
		cardsDisplay: true
	}),
	props: ['datasets', 'selectedType'],
	computed: {
		sortedList() {
			return this.datasets.filter(item => {
				return item.name.toLowerCase().includes(this.search.toLowerCase())
			})
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
			max-width: 426px;
		}
		&--sort {
			display: flex;
			gap: 20px;
			align-items: center;
			p {
				color: #A7BED3;
				font-size: 14px;
				* {
					vertical-align: middle;
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
			tr {
				height: 35px;
			}
			th {
				font-weight: inherit;
				padding: 0 50px;
				min-width: 150px;
				&:first-child {
					padding: 15px 10px;
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